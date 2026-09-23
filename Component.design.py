from collections import deque

import numpy as np
import pandas as pd
import yfinance as yf

class DataHandler:
    def __init__(self, ticker, start, end):
        raw_data = yf.download(ticker, start, end, interval="1d", auto_adjust=True)

        if raw_data.empty:
            raise ValueError (f"No data returned for {ticker}.")
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)

        self._data = raw_data[["Open", "High", "Low", "Close", "Volume"]]
        self._dates = raw_data.index
        self._open = raw_data["Open"].to_numpy()
        self._high = raw_data["High"].to_numpy()
        self._low = raw_data["Low"].to_numpy()
        self._close = raw_data["Close"].to_numpy()
        self._volume = raw_data["Volume"].to_numpy()

        self.current_index = -1

    def current_date(self):
        if self.current_index < 0:
            return None
        return self._dates[self.current_index]

    def advance(self):
        if self.current_index + 1 < len(self._dates):
            self.current_index += 1
            return True
        return False

    def get_current_bar(self):
        if self.current_index < 0:
            raise RuntimeError("No bar released yet, call advance() first.")
        i = self.current_index
        return {
            "date": self._dates[i],
            "open": float(self._open[i]),
            "high": float(self._high[i]),
            "low": float(self._low[i]),
            "close": float(self._close[i]),
            "volume": float(self._volume[i]),
        }

    def get_data_up_to_current(self):
        return self._data.iloc[:self.current_index + 1].copy()

    def reset(self):
        self.current_index = -1

    def stream(self):
        self.reset()
        while self.advance():
            yield self.get_current_bar()

class WilderAverage:
    def __init__(self, period):
        self.period = period
        self._seed_values = []
        self.value = None

    def update(self, x):
        if self.value is None:
            self._seed_values.append(x)
            if len(self._seed_values) == self.period:
                self.value = sum(self._seed_values) / self.period
        else:
            self.value = self.value + (x - self.value) / self.period
        return self.value

class EMA:
    def __init__(self, period):
        self.period = period
        self.smoothing_constant = 2 / (period + 1)
        self._seed_values = []
        self.value = None

    def update(self, x):
        if x is None:
            return self.value
        if self.value is None:
            self._seed_values.append(x)
            if len(self._seed_values) == self.period:
                self.value = sum(self._seed_values) / self.period
        else:
            self.value = x * self.smoothing_constant + self.value * (1 - self.smoothing_constant)
        return self.value

def true_range(high, low, prev_close):
    current_trading_range = high - low
    upward_gap = abs(high - prev_close)
    downward_gap = abs(low - prev_close)
    return max(current_trading_range, upward_gap, downward_gap)

class RegimeDetector:
    def __init__(self, period = 14, strength_threshold = 25, vol_window = 20, vol_lookback = 100, pct_threshold = 0.7):
        self.period = period
        self.strength_threshold = strength_threshold
        self.pct_threshold = pct_threshold

        self._prev_close = None
        self._prev_high = None
        self._prev_low = None

        self._smoothed_TR = WilderAverage(period)
        self._smoothed_pos_dm = WilderAverage(period)
        self._smoothed_neg_dm = WilderAverage(period)
        self._adx = WilderAverage(period)

        self._returns = deque(maxlen=vol_window)
        self._volatilities = deque(maxlen=vol_lookback)

        self.adx = None
        self.trend = None
        self.rolling_volatility = None
        self.volatility_percentile = None
        self.volatility_classification = None
        self.regime = None

    def update(self, bar):
        high, low, close = bar["high"], bar["low"], bar["close"]

        if self._prev_close is not None:
            self._update_trend_strength(high, low)
            self._update_volatility(close)

        self._prev_close = close
        self._prev_high = high
        self._prev_low = low

        self.trend = self.classify_trend(self.adx)
        self.volatility_classification = self.classify_volatility(self.volatility_percentile)
        self.regime = self.classify_regime(self.trend, self.volatility_classification)

        return {
            "adx": self.adx,
            "trend": self.trend,
            "rolling_volatility": self.rolling_volatility,
            "volatility_percentile": self.volatility_percentile,
            "volatility_classification": self.volatility_classification,
            "regime": self.regime,
        }

    def _update_trend_strength(self, high, low):
        tr = true_range(high, low, self._prev_close)

        up_move = high - self._prev_high
        down_move = self._prev_low - low

        if up_move > down_move and up_move > 0:
            pos_dm = up_move
        else:
            pos_dm = 0
        if down_move > up_move and down_move > 0:
            neg_dm = down_move
        else:
            neg_dm = 0

        smoothed_TR = self._smoothed_TR.update(tr)
        smoothed_pos_dm = self._smoothed_pos_dm.update(pos_dm)
        smoothed_neg_dm = self._smoothed_neg_dm.update(neg_dm)

        if smoothed_TR is None:
            return
        if smoothed_TR == 0:
            pos_di = 0
            neg_di = 0
        else:
            pos_di = 100 * smoothed_pos_dm / smoothed_TR
            neg_di = 100 * smoothed_neg_dm / smoothed_TR

        if pos_di == 0 and neg_di == 0:
            d_index = 0
        else:
            d_index = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)

        self.adx = self._adx.update(d_index)

    def _update_volatility(self, close):
        if self._prev_close == 0:
            return
        self._returns.append(close / self._prev_close - 1)

        if len(self._returns) < self._returns.maxlen:
            return
        self.rolling_volatility = float(np.std(self._returns, ddof=1))
        self._volatilities.append(self.rolling_volatility)

        if len(self._volatilities) < self._volatilities.maxlen:
            return
        below = sum(1 for v in self._volatilities if v < self.rolling_volatility)
        equal = sum(1 for v in self._volatilities if v == self.rolling_volatility)
        self.volatility_percentile = (below + (equal + 1) / 2) / len(self._volatilities)

    def classify_trend(self, adx):
        if adx is None:
            return None
        elif adx >= self.strength_threshold:
            return "trending"
        else:
            return "ranging"

    def classify_volatility(self, volatility_percentile):
        if volatility_percentile is None:
            return None
        elif volatility_percentile >= self.pct_threshold:
            return "high_vol"
        else:
            return "low_vol"

    def classify_regime(self, trend, volatility_classification):
        if trend is None or volatility_classification is None:
            return None
        return f"{trend}_{volatility_classification}"

class MomentumEntryIndicators:
    def __init__(self, atr_period = 14, donchian_period = 20, roc_period = 10, adx_slope_lookback = 5):
        self._prev_close = None

        self._atr = WilderAverage(atr_period)

        self._donchian_highs = deque(maxlen=donchian_period)
        self._donchian_lows = deque(maxlen=donchian_period)

        self._fast_ema = EMA(12)
        self._slow_ema = EMA(26)
        self._signal_line = EMA(9)

        self._roc_closes = deque(maxlen=roc_period + 1)

        self._adx_history = deque(maxlen=adx_slope_lookback + 1)

        self.atr = None
        self.donchian_high = None
        self.donchian_low = None
        self.macd = None
        self.signal_line = None
        self.macd_histogram = None
        self.rate_of_change = None
        self.adx_slope = None

    def update(self, bar, adx):
        high, low, close = bar["high"], bar["low"], bar["close"]

        self._update_atr(high, low)
        self._update_donchian(high, low)
        self._update_macd(close)
        self._update_rate_of_change(close)
        self._update_adx_slope(adx)

        self._prev_close = close

        return {
            "atr": self.atr,
            "donchian_high": self.donchian_high,
            "donchian_low": self.donchian_low,
            "macd": self.macd,
            "signal_line": self.signal_line,
            "macd_histogram": self.macd_histogram,
            "rate_of_change": self.rate_of_change,
            "adx_slope": self.adx_slope,
        }

    def _update_atr(self, high, low):
        if self._prev_close is None:
            return
        self.atr = self._atr.update(true_range(high, low, self._prev_close))

    def _update_donchian(self, high, low):
        if len(self._donchian_highs) == self._donchian_highs.maxlen:
            self.donchian_high = max(self._donchian_highs)
            self.donchian_low = min(self._donchian_lows)
        self._donchian_highs.append(high)
        self._donchian_lows.append(low)

    def _update_macd(self, close):
        fast = self._fast_ema.update(close)
        slow = self._slow_ema.update(close)

        if fast is None or slow is None:
            return
        self.macd = fast - slow
        self.signal_line = self._signal_line.update(self.macd)

        if self.signal_line is not None:
            self.macd_histogram = self.macd - self.signal_line

    def _update_rate_of_change(self, close):
        self._roc_closes.append(close)

        if len(self._roc_closes) < self._roc_closes.maxlen:
            return
        old_close = self._roc_closes[0]
        if old_close == 0: #extremely unlikely but why not hehe
            self.rate_of_change = None
        else:
            self.rate_of_change = (close - old_close) / old_close * 100

    def _update_adx_slope(self, adx):
        self._adx_history.append(adx)

        if len(self._adx_history) < self._adx_history.maxlen:
            return
        old_adx = self._adx_history[0]
        if adx is None or old_adx is None:
            self.adx_slope = None
        else:
            self.adx_slope = adx - old_adx

class MomentumSignals:
    pass

#testing

if __name__ == "__main__":
    stock = DataHandler("AAPL", "2025-01-01", "2026-01-20")
    regime = RegimeDetector()
    indicators = MomentumEntryIndicators()

    for bar in stock.stream():
        regime_values = regime.update(bar)
        indicator_values = indicators.update(bar, regime_values["adx"])
        print(bar["date"].date(), regime_values["regime"], indicator_values["macd_histogram"])
