import itertools
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf

PRICE_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_cache")

def download_prices(ticker, start, end, cache_dir = PRICE_CACHE_DIR):
    #cached so walk-forward and repeated runs don't hit yfinance every time (delete the file to refresh)
    cache_path = os.path.join(cache_dir, f"{ticker}_{start}_{end}.pkl")
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)

    raw_data = yf.download(ticker, start, end, interval="1d", auto_adjust=True)

    if raw_data.empty:
        raise ValueError (f"No data returned for {ticker}.")
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)

    prices = raw_data[["Open", "High", "Low", "Close", "Volume"]]
    os.makedirs(cache_dir, exist_ok=True)
    prices.to_pickle(cache_path)
    return prices

class DataHandler:
    def __init__(self, ticker, start, end):
        self._load(download_prices(ticker, start, end))

    @classmethod
    def from_frame(cls, prices):
        handler = cls.__new__(cls)
        handler._load(prices)
        return handler

    def _load(self, prices):
        self._data = prices
        self._dates = prices.index
        self._open = prices["Open"].to_numpy()
        self._high = prices["High"].to_numpy()
        self._low = prices["Low"].to_numpy()
        self._close = prices["Close"].to_numpy()
        self._volume = prices["Volume"].to_numpy()

        self.current_index = -1

    def get_dates(self):
        return self._dates

    def slice(self, start, end, warmup_bars = 0):
        #new handler over [start, end], plus warmup_bars bars before start so indicators are already warm at start
        start_index = self._dates.searchsorted(pd.Timestamp(start))
        end_index = self._dates.searchsorted(pd.Timestamp(end), side="right")
        return DataHandler.from_frame(self._data.iloc[max(0, start_index - warmup_bars):end_index])

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

    def get_close_series(self):
        #full history, only for after-the-fact reporting (benchmarks), never inside the backtest loop
        return self._data["Close"].copy()

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

class ADX:
    #split out of RegimeDetector so the weekly filter can reuse it on weekly bars
    def __init__(self, period = 14):
        self._prev_close = None
        self._prev_high = None
        self._prev_low = None

        self._smoothed_TR = WilderAverage(period)
        self._smoothed_pos_dm = WilderAverage(period)
        self._smoothed_neg_dm = WilderAverage(period)
        self._adx = WilderAverage(period)

        self.pos_di = None
        self.neg_di = None
        self.value = None

    def update(self, high, low, close):
        if self._prev_close is not None:
            self._update_trend_strength(high, low)

        self._prev_close = close
        self._prev_high = high
        self._prev_low = low
        return self.value

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

        self.pos_di = pos_di
        self.neg_di = neg_di
        self.value = self._adx.update(d_index)

class RegimeDetector:
    def __init__(self, period = 14, strength_threshold = 25, vol_window = 20, vol_lookback = 100, pct_threshold = 0.7):
        self.period = period
        self.strength_threshold = strength_threshold
        self.pct_threshold = pct_threshold

        self._prev_close = None
        self._adx = ADX(period)

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

        self.adx = self._adx.update(high, low, close)
        if self._prev_close is not None:
            self._update_volatility(close)
        self._prev_close = close

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
    #use_regime_filter / use_ensemble exist for the ablation study, switching one off shows what it contributes
    def __init__(self, regime, indicators, vote_threshold = 3,
                 roc_min_threshold = 2.0, macd_slope_lookback = 3,
                 use_regime_filter = True, use_ensemble = True):
        self.regime = regime
        self.indicators = indicators
        self.use_regime_filter = use_regime_filter
        self.use_ensemble = use_ensemble
        self.vote_threshold = vote_threshold
        self.roc_min_threshold = roc_min_threshold
        self.macd_slope_lookback = macd_slope_lookback
        self._macd_histogram_history = deque(maxlen=macd_slope_lookback)

    def donchian_breakout_vote(self, bar):
        donchian_high = self.indicators.donchian_high

        if donchian_high is None:
            return False

        return bar["close"] > donchian_high

    def macd_slope_vote(self):
        histogram = self.indicators.macd_histogram

        if histogram is None:
            return False

        self._macd_histogram_history.append(histogram)

        if len(self._macd_histogram_history) < self.macd_slope_lookback:
            return False

        history = list(self._macd_histogram_history)
        is_rising = all(history[i] < history[i+1] for i in range(len(history) -1))

        return histogram > 0 and is_rising

    def roc_vote(self):
        roc = self.indicators.rate_of_change

        if roc is None:
            return False

        return roc >= self.roc_min_threshold

    def adx_slope_vote(self):
        adx_slope = self.indicators.adx_slope

        if adx_slope is None:
            return False

        return adx_slope > 0 

    def regime_allows_entry(self):
        regime = self.regime.regime

        if regime is None:
            return False
        if not self.use_regime_filter:
            return True

        return regime.startswith("trending")

    def generate_signal(self, bar):
        votes = {
            "donchian": self.donchian_breakout_vote(bar),
            "macd_slope": self.macd_slope_vote(),
            "roc": self.roc_vote(),
            "adx_slope": self.adx_slope_vote(),
        }
        score = sum(votes.values())
        regime_ok = self.regime_allows_entry()

        if self.use_ensemble:
            votes_ok = score >= self.vote_threshold
        else:
            votes_ok = votes["donchian"] #a plain donchian breakout, the classic single-signal entry

        return {
            "entry": regime_ok and votes_ok,
            "score": score,
            "votes": votes,
            "regime": self.regime.regime,
        }

class WeeklyTrendFilter:
    #builds weekly bars from the daily stream. a week only counts once the first bar of the next week arrives,
    #so the in progress week's high/low/close never leak in (costs a day of lag on fridays, but holidays make "is this the last day of the week" unknowable in real time)
    def __init__(self, period = 14, strength_threshold = 25, require_uptrend = True):
        self.strength_threshold = strength_threshold
        self.require_uptrend = require_uptrend
        self._adx = ADX(period)

        self._week = None
        self._week_high = None
        self._week_low = None
        self._week_close = None

        self.completed_weeks = 0
        self.adx = None
        self.trend = None

    def update(self, bar):
        week = tuple(bar["date"].isocalendar())[:2]

        if week != self._week:
            if self._week is not None:
                self._complete_week()
            self._week = week
            self._week_high = bar["high"]
            self._week_low = bar["low"]
        else:
            self._week_high = max(self._week_high, bar["high"])
            self._week_low = min(self._week_low, bar["low"])
        self._week_close = bar["close"]

        return self.trend

    def _complete_week(self):
        self.adx = self._adx.update(self._week_high, self._week_low, self._week_close)
        self.completed_weeks += 1

        if self.adx is None:
            self.trend = None
        elif self.adx >= self.strength_threshold:
            self.trend = "trending"
        else:
            self.trend = "ranging"

    def allows_entry(self):
        if self.trend != "trending":
            return False
        #ADX alone has no direction, so a strong weekly downtrend would also read "trending"
        if self.require_uptrend:
            return self._adx.pos_di > self._adx.neg_di
        return True

class MomentumCrashFilter:
    #flags a volatility spike right after a long calm uptrend, the setup that tends to precede momentum crashes
    def __init__(self, min_low_vol_streak = 20, jump_threshold = 0.30, window = 5, cooldown_days = 10):
        self.min_low_vol_streak = min_low_vol_streak
        self.jump_threshold = jump_threshold
        self.cooldown_days = cooldown_days

        self._low_vol_streak = 0
        self._recent_percentiles = deque(maxlen=window + 1)
        self._recent_streaks = deque(maxlen=window + 1)

        self.triggered = False
        self.cooldown_remaining = 0
        self.trigger_dates = []

    def update(self, date, regime, volatility_percentile):
        if regime == "trending_low_vol":
            self._low_vol_streak += 1
        else:
            self._low_vol_streak = 0

        self.triggered = False
        if volatility_percentile is not None:
            self._recent_percentiles.append(volatility_percentile)
            self._recent_streaks.append(self._low_vol_streak)

            jump = volatility_percentile - min(self._recent_percentiles)
            #the spike itself usually flips the regime to high vol and resets the streak, so look back over the window
            after_calm_trend = max(self._recent_streaks) >= self.min_low_vol_streak
            self.triggered = jump > self.jump_threshold and after_calm_trend

        if self.triggered:
            if self.cooldown_remaining == 0:
                self.trigger_dates.append(date)
            self.cooldown_remaining = self.cooldown_days
        elif self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        return self.triggered

    def blocks_entry(self):
        return self.cooldown_remaining > 0

class ATRTrailingStop:
    #k is locked in at entry from the regime at that time, so the stop doesn't jump if the regime flips mid-trade
    #regime_based = False uses one k (the midpoint) everywhere, to test whether the regime adds anything to the stop
    def __init__(self, k_low_vol = 2.0, k_high_vol = 3.5, regime_based = True):
        self.k_low_vol = k_low_vol
        self.k_high_vol = k_high_vol
        self.regime_based = regime_based

        self.k = None
        self.highest_close = None
        self.level = None

    def k_for_regime(self, regime):
        if not self.regime_based:
            return (self.k_low_vol + self.k_high_vol) / 2
        if regime is not None and regime.endswith("high_vol"):
            return self.k_high_vol
        return self.k_low_vol

    def start(self, entry_price, k, atr):
        self.k = k
        self.highest_close = entry_price
        self.level = entry_price - k * atr

    def update(self, close, atr):
        if self.level is None or atr is None:
            return self.level

        self.highest_close = max(self.highest_close, close)
        #uses today's ATR (chandelier style), the max() ratchet stops a volatility spike from lowering it
        self.level = max(self.level, self.highest_close - self.k * atr)
        return self.level

    def is_hit(self, close):
        return self.level is not None and close < self.level

    def reset(self):
        self.k = None
        self.highest_close = None
        self.level = None

class Portfolio:
    def __init__(self, starting_cash = 10000, cost_bps = 5):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.cost_rate = cost_bps / 10000 #charged per side, so a round trip costs 2x

        self.shares = 0
        self.entry_price = None
        self.entry_date = None
        self._entry_cost = 0
        self._entry_info = {}

        self.trades = []
        self.equity_curve = []

    def is_flat(self):
        return self.shares == 0

    def max_affordable_shares(self, price):
        return int(self.cash // (price * (1 + self.cost_rate)))

    def buy(self, date, price, shares, entry_info = None):
        if shares <= 0 or not self.is_flat():
            return False

        notional = shares * price
        cost = notional * self.cost_rate
        if notional + cost > self.cash:
            raise ValueError(f"Not enough cash to buy {shares} shares at {price:.2f}.")

        self.cash -= notional + cost
        self.shares = shares
        self.entry_price = price
        self.entry_date = date
        self._entry_cost = cost
        self._entry_info = entry_info or {}
        return True

    def sell(self, date, price, reason):
        if self.is_flat():
            return None

        notional = self.shares * price
        exit_cost = notional * self.cost_rate
        self.cash += notional - exit_cost

        total_cost = self._entry_cost + exit_cost
        pnl = (price - self.entry_price) * self.shares - total_cost
        trade = {
            "entry_date": self.entry_date,
            "entry_price": self.entry_price,
            "exit_date": date,
            "exit_price": price,
            "shares": self.shares,
            "pnl": pnl,
            "return_pct": pnl / (self.entry_price * self.shares) * 100,
            "costs": total_cost,
            "exit_reason": reason,
            **self._entry_info,
        }
        self.trades.append(trade)

        self.shares = 0
        self.entry_price = None
        self.entry_date = None
        self._entry_cost = 0
        self._entry_info = {}
        return trade

    def equity(self, price):
        return self.cash + self.shares * price

    def record_equity(self, date, price):
        self.equity_curve.append({
            "date": date,
            "equity": self.equity(price),
            "cash": self.cash,
            "shares": self.shares,
        })

class Backtester:
    #signals are computed on day t's close, orders fill at day t+1's open to avoid lookahead
    #weekly_filter and crash_filter are optional, pass None to switch a component off
    def __init__(self, data, regime, indicators, signals, portfolio, stop,
                 weekly_filter = None, crash_filter = None,
                 risk_pct = 0.01, max_position_pct = 1.0, trade_start = None, use_trailing_stop = True):
        self.data = data
        self.regime = regime
        self.indicators = indicators
        self.signals = signals
        self.portfolio = portfolio
        self.stop = stop
        self.weekly_filter = weekly_filter
        self.crash_filter = crash_filter
        self.risk_pct = risk_pct
        self.max_position_pct = max_position_pct
        #bars before trade_start still update every component (warm-up) but never trade or record equity
        self.trade_start = pd.Timestamp(trade_start) if trade_start is not None else None
        self.use_trailing_stop = use_trailing_stop

        self._pending_order = None
        self.blocked_entries = {"weekly_filter": 0, "crash_cooldown": 0}

    def run(self):
        last_bar = None

        for bar in self.data.stream():
            last_bar = bar
            self._fill_pending_order(bar)

            regime_values = self.regime.update(bar)
            self.indicators.update(bar, regime_values["adx"])
            #called every bar, even while in a position, so the MACD slope history stays contiguous
            signal = self.signals.generate_signal(bar)

            #filters also update every bar so their state (weekly bars, low vol streak) is never stale
            if self.weekly_filter is not None:
                self.weekly_filter.update(bar)
            crash = False
            if self.crash_filter is not None:
                crash = self.crash_filter.update(bar["date"], regime_values["regime"],
                                                 regime_values["volatility_percentile"])

            if regime_values["regime"] is None: #still warming up, no trading and no equity recorded
                continue
            if self.trade_start is not None and bar["date"] < self.trade_start:
                continue

            atr = self.indicators.atr

            #the crash filter overrides everything else
            if crash and not self.portfolio.is_flat():
                self._pending_order = {"side": "sell", "reason": "crash_filter"}
            elif not self.portfolio.is_flat():
                exit_reason = self.check_exit(bar, atr)
                if exit_reason is not None:
                    self._pending_order = {"side": "sell", "reason": exit_reason}
            elif signal["entry"] and atr and self._entry_allowed():
                self._pending_order = {
                    "side": "buy",
                    "reason": "entry_signal",
                    "k": self.stop.k_for_regime(regime_values["regime"]),
                    "atr": atr,
                    "regime": regime_values["regime"],
                }

            self.portfolio.record_equity(bar["date"], bar["close"])

        #no next open to fill at, so any open position is closed at the final close
        if last_bar is not None and not self.portfolio.is_flat():
            self.portfolio.sell(last_bar["date"], last_bar["close"], "end_of_data")
            self.stop.reset()
            self.portfolio.equity_curve.pop()
            self.portfolio.record_equity(last_bar["date"], last_bar["close"])
        self._pending_order = None

        return self.results()

    def _entry_allowed(self):
        #counts which filter vetoed an otherwise valid daily entry signal
        if self.crash_filter is not None and self.crash_filter.blocks_entry():
            self.blocked_entries["crash_cooldown"] += 1
            return False
        if self.weekly_filter is not None and not self.weekly_filter.allows_entry():
            self.blocked_entries["weekly_filter"] += 1
            return False
        return True

    def _fill_pending_order(self, bar):
        order = self._pending_order
        self._pending_order = None

        if order is None:
            return
        if order["side"] == "buy":
            price = bar["open"]
            shares = self.size_position(price, order["k"], order["atr"])
            entry_info = {"entry_regime": order["regime"], "k": order["k"], "entry_atr": order["atr"]}
            if self.portfolio.buy(bar["date"], price, shares, entry_info):
                self.stop.start(price, order["k"], order["atr"])
        elif order["side"] == "sell":
            self.portfolio.sell(bar["date"], bar["open"], order["reason"])
            self.stop.reset()

    def size_position(self, price, k, atr):
        #shares = risk% x equity / stop distance, so a stop-out loses roughly risk% of equity
        #k alone carries the regime adjustment: tight k in low vol -> bigger size, wide k in high vol -> smaller
        equity = self.portfolio.equity(price)
        risk_shares = int(self.risk_pct * equity // (k * atr))

        #cap the position value so quiet, low ATR stocks can't produce a leveraged position
        cap_shares = int(self.max_position_pct * equity // price)

        return min(risk_shares, cap_shares, self.portfolio.max_affordable_shares(price))

    def check_exit(self, bar, atr):
        if not self.use_trailing_stop:
            #simple exit for the ablation study: a close below the prior donchian low
            donchian_low = self.indicators.donchian_low
            if donchian_low is not None and bar["close"] < donchian_low:
                return "donchian_low"
            return None

        self.stop.update(bar["close"], atr)

        if self.stop.is_hit(bar["close"]):
            return "trailing_stop"
        return None

    def results(self):
        equity_curve = pd.DataFrame(self.portfolio.equity_curve)
        if not equity_curve.empty:
            equity_curve = equity_curve.set_index("date")

        return {
            "equity_curve": equity_curve,
            "trades": pd.DataFrame(self.portfolio.trades),
            "blocked_entries": dict(self.blocked_entries),
            "crash_triggers": list(self.crash_filter.trigger_dates) if self.crash_filter is not None else [],
        }

DEFAULT_PARAMS = {
    "starting_cash": 10000,
    "cost_bps": 5,
    "donchian_period": 20,
    "roc_period": 10,
    "roc_min_threshold": 2.0,
    "macd_slope_lookback": 3,
    "vote_threshold": 3,
    "k_low_vol": 2.0,
    "k_high_vol": 3.5,
    "risk_pct": 0.01,
    "max_position_pct": 1.0,
    "use_weekly_filter": False, #tested and rejected: lowered out-of-sample sharpe on 15 of 18 tickers
    "weekly_adx_threshold": 25,
    "use_crash_filter": True,
    "crash_min_low_vol_streak": 20,
    "crash_jump_threshold": 0.30,
    "crash_window": 5,
    "crash_cooldown_days": 10,
    "adx_threshold": 25,
    "use_regime_filter": False, #tested and rejected as an entry gate: entering in any regime beat it on 13 of 18 tickers
    "regime_stop_width": True, #the regime's remaining job: tighter stop in low vol, wider in high vol
    "use_ensemble": True,
    "use_trailing_stop": True,
}

def run_backtest(data, params = None, trade_start = None):
    #fresh components every run, so the same DataHandler can be backtested many times
    params = {**DEFAULT_PARAMS, **(params or {})}

    regime = RegimeDetector(strength_threshold = params["adx_threshold"])
    indicators = MomentumEntryIndicators(donchian_period = params["donchian_period"],
                                         roc_period = params["roc_period"])
    signals = MomentumSignals(regime, indicators,
                              vote_threshold = params["vote_threshold"],
                              roc_min_threshold = params["roc_min_threshold"],
                              macd_slope_lookback = params["macd_slope_lookback"],
                              use_regime_filter = params["use_regime_filter"],
                              use_ensemble = params["use_ensemble"])
    portfolio = Portfolio(params["starting_cash"], params["cost_bps"])
    stop = ATRTrailingStop(params["k_low_vol"], params["k_high_vol"], regime_based = params["regime_stop_width"])

    weekly_filter = None
    if params["use_weekly_filter"]:
        weekly_filter = WeeklyTrendFilter(strength_threshold = params["weekly_adx_threshold"])
    crash_filter = None
    if params["use_crash_filter"]:
        crash_filter = MomentumCrashFilter(min_low_vol_streak = params["crash_min_low_vol_streak"],
                                           jump_threshold = params["crash_jump_threshold"],
                                           window = params["crash_window"],
                                           cooldown_days = params["crash_cooldown_days"])

    backtest = Backtester(data, regime, indicators, signals, portfolio, stop,
                          weekly_filter = weekly_filter, crash_filter = crash_filter,
                          risk_pct = params["risk_pct"], max_position_pct = params["max_position_pct"],
                          trade_start = trade_start, use_trailing_stop = params["use_trailing_stop"])
    results = backtest.run()
    results["params"] = params
    return results

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0 #annual, kept at zero so sharpe/sortino are plain return-to-risk ratios

def equity_metrics(equity, risk_free_rate = RISK_FREE_RATE, periods_per_year = TRADING_DAYS_PER_YEAR):
    returns = equity.pct_change().dropna()
    excess = returns - risk_free_rate / periods_per_year
    years = (equity.index[-1] - equity.index[0]).days / 365.25

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
    volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)

    excess_std = excess.std(ddof=1)
    sharpe = excess.mean() / excess_std * np.sqrt(periods_per_year) if excess_std > 0 else np.nan

    #downside deviation uses every day (0 on up days), not just the down days
    downside_dev = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    sortino = excess.mean() / downside_dev * np.sqrt(periods_per_year) if downside_dev > 0 else np.nan

    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }

def trade_metrics(trades, equity_curve):
    time_in_market = (equity_curve["shares"] > 0).mean()

    if trades.empty:
        return {"trades": 0, "win_rate": np.nan, "avg_win": np.nan, "avg_loss": np.nan,
                "payoff_ratio": np.nan, "profit_factor": np.nan, "avg_bars_held": np.nan,
                "time_in_market": time_in_market, "total_costs": 0.0}

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]

    avg_win = wins["return_pct"].mean() / 100 if not wins.empty else np.nan
    avg_loss = losses["return_pct"].mean() / 100 if not losses.empty else np.nan
    gross_profit = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()

    dates = equity_curve.index
    bars_held = [dates.get_loc(exit_date) - dates.get_loc(entry_date)
                 for entry_date, exit_date in zip(trades["entry_date"], trades["exit_date"])]

    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": avg_win / abs(avg_loss) if avg_loss and not np.isnan(avg_loss) else np.nan,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.nan,
        "avg_bars_held": float(np.mean(bars_held)),
        "time_in_market": time_in_market,
        "total_costs": trades["costs"].sum(),
    }

def buy_and_hold_equity(closes, starting_cash):
    #frictionless with fractional shares, which slightly flatters the benchmark
    return starting_cash * closes / closes.iloc[0]

def performance_report(data, named_results):
    #one column per backtest, plus buy & hold over the first backtest's dates
    columns = {}
    for name, results in named_results.items():
        equity_curve = results["equity_curve"]
        columns[name] = {**equity_metrics(equity_curve["equity"]),
                         **trade_metrics(results["trades"], equity_curve)}

    first = next(iter(named_results.values()))
    closes = data.get_close_series().loc[first["equity_curve"].index]
    columns["Buy & Hold"] = {**equity_metrics(buy_and_hold_equity(closes, first["params"]["starting_cash"])),
                             "time_in_market": 1.0}

    return pd.DataFrame(columns)

PERCENT_METRICS = {"total_return", "cagr", "volatility", "max_drawdown",
                   "win_rate", "avg_win", "avg_loss", "time_in_market"}

def format_report(report):
    def fmt(metric, value):
        if pd.isna(value):
            return "-"
        if metric in PERCENT_METRICS:
            return f"{value * 100:.2f}%"
        if metric == "trades":
            return f"{int(value)}"
        if metric == "total_costs":
            return f"${value:,.2f}"
        if metric == "sharpe_change":
            return f"{value:+.2f}"
        return f"{value:.2f}"

    return report.apply(lambda column: [fmt(metric, value) for metric, value in column.items()])

#each axis is a list of values for one param, or a list of dicts for params that move together (the two stop multipliers)
#kept small on purpose: 3 x 3 x 2 x 3 = 54 backtests per window
#risk_pct is left out: sharpe barely changes with position size, so picking it by sharpe would be meaningless
PARAM_GRID = {
    "vote_threshold": [2, 3, 4],
    "donchian_period": [10, 20, 40],
    "roc_min_threshold": [0.0, 3.0],
    "stop_k": [{"k_low_vol": 1.5, "k_high_vol": 2.5},
               {"k_low_vol": 2.0, "k_high_vol": 3.5},
               {"k_low_vol": 3.0, "k_high_vol": 5.0}],
}

def expand_grid(param_grid):
    combos = []
    for values in itertools.product(*param_grid.values()):
        params = {}
        for key, value in zip(param_grid, values):
            if isinstance(value, dict):
                params.update(value)
            else:
                params[key] = value
        combos.append(params)
    return combos

def walk_forward_windows(dates, train_years = 3, test_years = 1, warmup_bars = 252):
    #rolling windows: train on train_years, test on the next test_years, then slide forward by test_years
    #the first warmup_bars of data are reserved so even the first training window starts with warm indicators
    windows = []
    train_start = dates[warmup_bars]

    while True:
        test_start = train_start + pd.DateOffset(years = train_years)
        if test_start > dates[-1] - pd.Timedelta(days = 30): #skip a final test window too short to mean anything
            break
        test_end = min(test_start + pd.DateOffset(years = test_years) - pd.Timedelta(days = 1), dates[-1])

        windows.append({
            "train_start": train_start,
            "train_end": test_start - pd.Timedelta(days = 1),
            "test_start": test_start,
            "test_end": test_end,
        })
        train_start = train_start + pd.DateOffset(years = test_years)

    return windows

def select_params(data, window, param_grid, base_params, warmup_bars, min_trades):
    train_data = data.slice(window["train_start"], window["train_end"], warmup_bars)
    best = None

    for grid_params in expand_grid(param_grid):
        params = {**base_params, **grid_params}
        results = run_backtest(train_data, params, trade_start = window["train_start"])

        #a great sharpe from two trades is luck, not skill
        if len(results["trades"]) < min_trades:
            continue
        sharpe = equity_metrics(results["equity_curve"]["equity"])["sharpe"]
        if np.isnan(sharpe):
            continue

        if best is None or sharpe > best["train_sharpe"]:
            best = {"params": params, "train_sharpe": sharpe, "train_trades": len(results["trades"])}

    return best

def stitch_equity_curves(segments, starting_cash):
    #each test segment starts flat with fresh cash, so chain their daily returns rather than their dollar values
    stitched = pd.concat(segments)[["shares"]]
    returns = pd.concat([segment["equity"].pct_change().fillna(0) for segment in segments])
    stitched["equity"] = starting_cash * (1 + returns).cumprod()
    return stitched

def walk_forward(data, param_grid = PARAM_GRID, base_params = None, train_years = 3, test_years = 1,
                 warmup_bars = 252, min_trades = 6, verbose = True):
    base_params = {**DEFAULT_PARAMS, **(base_params or {})}
    windows = walk_forward_windows(data.get_dates(), train_years, test_years, warmup_bars)
    grid_keys = list(expand_grid(param_grid)[0])

    segments, segments_no_costs = [], []
    trades, trades_no_costs = [], []
    summary = []

    for number, window in enumerate(windows, 1):
        best = select_params(data, window, param_grid, base_params, warmup_bars, min_trades)
        if best is None:
            #nothing traded often enough in training, so fall back to the defaults rather than trust a thin sample
            best = {"params": base_params, "train_sharpe": np.nan, "train_trades": 0}
        params = best["params"]

        #parameters are frozen here, the test window has never been seen by the grid search
        test_data = data.slice(window["test_start"], window["test_end"], warmup_bars)
        test = run_backtest(test_data, params, trade_start = window["test_start"])
        test_no_costs = run_backtest(test_data, {**params, "cost_bps": 0}, trade_start = window["test_start"])

        segments.append(test["equity_curve"])
        segments_no_costs.append(test_no_costs["equity_curve"])
        trades.append(test["trades"])
        trades_no_costs.append(test_no_costs["trades"])

        test_metrics = equity_metrics(test["equity_curve"]["equity"])
        summary.append({
            "test_start": window["test_start"].date(),
            "test_end": window["test_end"].date(),
            "train_sharpe": best["train_sharpe"],
            "train_trades": best["train_trades"],
            "test_return": test_metrics["total_return"],
            "test_sharpe": test_metrics["sharpe"],
            "test_trades": len(test["trades"]),
            **{key: params[key] for key in grid_keys},
        })

        if verbose:
            print(f"[{number}/{len(windows)}] test {window['test_start'].date()} to {window['test_end'].date()}: "
                  f"train sharpe {best['train_sharpe']:.2f} ({best['train_trades']} trades), "
                  f"test return {test_metrics['total_return'] * 100:.2f}% ({len(test['trades'])} trades)", flush=True)

    def combine(trade_logs):
        trade_logs = [log for log in trade_logs if not log.empty]
        return pd.concat(trade_logs, ignore_index=True) if trade_logs else pd.DataFrame()

    starting_cash = base_params["starting_cash"]
    return {
        "equity_curve": stitch_equity_curves(segments, starting_cash),
        "trades": combine(trades),
        "params": base_params,
        "windows": pd.DataFrame(summary),
        "no_costs": {
            "equity_curve": stitch_equity_curves(segments_no_costs, starting_cash),
            "trades": combine(trades_no_costs),
            "params": base_params,
        },
    }

def in_sample_optimized(data, start, end, param_grid = PARAM_GRID, warmup_bars = 252, min_trades_per_year = 2):
    #the number walk-forward guards against: the best grid point picked with hindsight over the whole period
    window = {"train_start": pd.Timestamp(start), "train_end": pd.Timestamp(end)}
    years = (window["train_end"] - window["train_start"]).days / 365.25
    best = select_params(data, window, param_grid, DEFAULT_PARAMS, warmup_bars, int(min_trades_per_year * years))
    params = best["params"] if best is not None else DEFAULT_PARAMS

    return run_backtest(data.slice(start, end, warmup_bars), params, trade_start = start)

#each variant switches exactly one component off relative to the full strategy, except the rejected
#components (regime entry gate, weekly filter), which are switched back on so the evidence for dropping them stays reproducible
ABLATIONS = {
    "Full strategy": {},
    "Fixed stop width (no regime)": {"regime_stop_width": False},
    "No ensemble (Donchian only)": {"use_ensemble": False},
    "No crash filter": {"use_crash_filter": False},
    "Simple exit (no trailing stop)": {"use_trailing_stop": False},
    "Add regime entry gate (rejected)": {"use_regime_filter": True},
    "Add weekly filter (rejected)": {"use_weekly_filter": True},
}

def ablation_study(data, start, end, ablations = ABLATIONS, base_params = None, warmup_bars = 252):
    #fixed default params (nothing optimized), so the differences come from the components rather than from fitting
    window = data.slice(start, end, warmup_bars)
    rows = {}
    for name, overrides in ablations.items():
        results = run_backtest(window, {**(base_params or {}), **overrides}, trade_start = start)
        equity_curve = results["equity_curve"]
        rows[name] = {**equity_metrics(equity_curve["equity"]), **trade_metrics(results["trades"], equity_curve)}

    table = pd.DataFrame(rows)
    table.loc["sharpe_change"] = table.loc["sharpe"] - table.loc["sharpe", "Full strategy"]
    return table

def regime_history(data, adx_threshold = DEFAULT_PARAMS["adx_threshold"]):
    #for chart shading only. regime params aren't in the grid, but window warm-ups mean a backtest's
    #ADX can differ very slightly from this single full-history pass
    detector = RegimeDetector(strength_threshold = adx_threshold)
    return pd.Series({bar["date"]: detector.update(bar)["regime"] for bar in data.stream()})

#light chart surface and ink, plus the first three slots of a colorblind-validated categorical palette
CHART_COLORS = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "strategy": "#2a78d6",
    "trending_high_vol": "#eb6834",
    "trending_low_vol": "#1baf7a",
}

def _style_axis(ax):
    ax.set_facecolor(CHART_COLORS["surface"])
    ax.grid(True, color=CHART_COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(CHART_COLORS["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=CHART_COLORS["muted"], labelsize=9)
    ax.yaxis.label.set_color(CHART_COLORS["ink_secondary"])

def _dollar_log_axis(ax):
    #plain dollar ticks at 1-2-5 steps instead of matplotlib's default "3 x 10^2" log labels
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda value, _: f"${value:,.0f}" if value >= 10 else f"${value:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

def _label_point(ax, date, value, text, offset = (6, 0)):
    ax.annotate(text, xy=(date, value), xytext=offset, textcoords="offset points",
                ha="left", va="center", fontsize=9, color=CHART_COLORS["ink_secondary"], annotation_clip=False,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": CHART_COLORS["surface"], "edgecolor": "none", "alpha": 0.85})

def plot_backtest(data, results, title, path = None, start = None, end = None):
    equity = results["equity_curve"]["equity"].loc[start:end]
    dates = equity.index
    closes = data.get_close_series().loc[dates]
    regimes = regime_history(data, results["params"]["adx_threshold"]).reindex(dates)
    benchmark = buy_and_hold_equity(closes, equity.iloc[0]) #indexed to the strategy's equity on day one

    trades = results["trades"]
    entries = trades[(trades["entry_date"] >= dates[0]) & (trades["entry_date"] <= dates[-1])] if not trades.empty else trades
    exits = trades[(trades["exit_date"] >= dates[0]) & (trades["exit_date"] <= dates[-1])] if not trades.empty else trades

    with plt.rc_context({"font.family": ["Segoe UI", "DejaVu Sans"]}):
        fig, (ax_price, ax_equity, ax_drawdown) = plt.subplots(
            3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.3]})
        fig.patch.set_facecolor(CHART_COLORS["surface"])

        #price with regime shading and trade markers
        for regime, label in [("trending_low_vol", "Trending, low vol"), ("trending_high_vol", "Trending, high vol")]:
            ax_price.fill_between(dates, 0, 1, where=(regimes == regime).to_numpy(), step="post",
                                  transform=ax_price.get_xaxis_transform(), color=CHART_COLORS[regime],
                                  alpha=0.16, linewidth=0, label=label)
        ax_price.plot(dates, closes, color=CHART_COLORS["ink"], linewidth=1.2, label="Close")
        if not entries.empty:
            ax_price.scatter(entries["entry_date"], entries["entry_price"], marker="^", s=55, zorder=3,
                             color=CHART_COLORS["strategy"], edgecolors=CHART_COLORS["surface"], linewidths=1.5, label="Entry")
        if not exits.empty:
            ax_price.scatter(exits["exit_date"], exits["exit_price"], marker="v", s=55, zorder=3,
                             color=CHART_COLORS["ink"], edgecolors=CHART_COLORS["surface"], linewidths=1.5, label="Exit")
        _dollar_log_axis(ax_price)
        ax_price.set_ylabel("Price (log scale)")
        ax_price.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5, frameon=False, fontsize=9,
                        labelcolor=CHART_COLORS["ink_secondary"])

        #equity, both lines on one log axis from the same starting value
        ax_equity.plot(dates, equity, color=CHART_COLORS["strategy"], linewidth=2, label="Strategy")
        ax_equity.plot(dates, benchmark, color=CHART_COLORS["muted"], linewidth=2, label="Buy & hold")
        _label_point(ax_equity, dates[-1], equity.iloc[-1], f"${equity.iloc[-1]:,.0f}")
        _label_point(ax_equity, dates[-1], benchmark.iloc[-1], f"${benchmark.iloc[-1]:,.0f}")
        _dollar_log_axis(ax_equity)
        ax_equity.set_ylabel("Equity (log scale)")
        ax_equity.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=CHART_COLORS["ink_secondary"])

        #drawdown from the running peak, same colors as the equity panel, worst point labelled directly
        for series, color in [(equity, CHART_COLORS["strategy"]), (benchmark, CHART_COLORS["muted"])]:
            drawdown = series / series.cummax() - 1
            ax_drawdown.fill_between(dates, drawdown, 0, color=color, alpha=0.15, linewidth=0)
            ax_drawdown.plot(dates, drawdown, color=color, linewidth=1.5)
            _label_point(ax_drawdown, drawdown.idxmin(), drawdown.min(), f"Max {drawdown.min():.1%}")
        ax_drawdown.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax_drawdown.set_ylabel("Drawdown")

        for ax in (ax_price, ax_equity, ax_drawdown):
            _style_axis(ax)
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=CHART_COLORS["ink"])
        fig.tight_layout()

        if path is not None:
            fig.savefig(path, dpi=150, facecolor=CHART_COLORS["surface"])
    return fig


REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
#starts in 2002 so that, after a year of warm-up and three of training, 2006 onwards (incl. 2008-09) is out of sample
DATA_START = "2002-01-01"
DATA_END = "2026-01-01"

def single_ticker_report(ticker, start = DATA_START, end = DATA_END, report_dir = REPORT_DIR):
    stock = DataHandler(ticker, start, end)
    os.makedirs(report_dir, exist_ok=True)

    started = time.perf_counter()
    wf = walk_forward(stock)
    print(f"Walk-forward finished in {time.perf_counter() - started:.0f}s\n")

    windows = wf["windows"].copy()
    windows["test_return"] = (windows["test_return"] * 100).map("{:.2f}%".format)
    print(windows.round(2).to_string(index=False))

    oos_start, oos_end = wf["equity_curve"].index[0], wf["equity_curve"].index[-1]
    print("\nOptimizing the same grid over the whole period with hindsight, for the in-sample comparison...")
    in_sample = in_sample_optimized(stock, oos_start, oos_end)

    report = performance_report(stock, {
        "Walk-forward (OOS)": wf,
        "Walk-forward, no costs": wf["no_costs"],
        "In-sample (hindsight)": in_sample,
    })
    print(f"\n{oos_start.date()} to {oos_end.date()}"
          f"   (annualized with {TRADING_DAYS_PER_YEAR} days, risk free rate {RISK_FREE_RATE:.0%})")
    print(format_report(report).to_string())
    cost_drag = report.loc["cagr", "Walk-forward, no costs"] - report.loc["cagr", "Walk-forward (OOS)"]
    print(f"Transaction cost drag: {cost_drag * 100:.3f} CAGR points per year")

    ablation = ablation_study(stock, oos_start, oos_end)
    ablation_rows = ["cagr", "volatility", "sharpe", "sharpe_change", "sortino", "max_drawdown", "calmar",
                     "trades", "win_rate", "profit_factor", "time_in_market"]
    print("\nAblation, default params, each row switches one component off (or adds back the rejected weekly filter):")
    print(format_report(ablation.loc[ablation_rows]).T.to_string())

    report.to_csv(os.path.join(report_dir, f"{ticker}_performance.csv"))
    ablation.to_csv(os.path.join(report_dir, f"{ticker}_ablation.csv"))
    wf["windows"].to_csv(os.path.join(report_dir, f"{ticker}_walk_forward_windows.csv"), index=False)
    wf["trades"].to_csv(os.path.join(report_dir, f"{ticker}_oos_trades.csv"), index=False)

    plot_backtest(stock, wf, f"{ticker} momentum strategy, walk-forward out-of-sample {oos_start.year} to {oos_end.year}",
                  path=os.path.join(report_dir, f"{ticker}_overview.png"))
    plot_backtest(stock, wf, f"{ticker} trade detail, 2019 to 2020 (out-of-sample)",
                  path=os.path.join(report_dir, f"{ticker}_detail_2019_2020.png"), start="2019-01-01", end="2020-12-31")
    print(f"\nCharts and tables saved to {report_dir}")

#a deliberate mix: index ETFs, today's mega-cap winners, and stocks that were large caps in 2002 but had mixed or bad
#runs since (GE, INTC, C...), so a good result can't just come from picking stocks we already know went up
UNIVERSE = {
    "Index ETF": ["SPY", "QQQ", "IWM"],
    "Mega-cap winner": ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"],
    "2002 large cap": ["GE", "INTC", "XOM", "JNJ", "KO", "PFE", "IBM", "C", "WMT", "JPM"],
}

#ablation variant -> (component, sign). removing a helpful component lowers sharpe (sign -1),
#adding a helpful one raises it (sign +1), so sign x sharpe_change is always "how much the component helps"
ABLATION_COMPONENTS = {
    "Fixed stop width (no regime)": ("Regime stop width", -1),
    "No ensemble (Donchian only)": ("Ensemble", -1),
    "No crash filter": ("Crash filter", -1),
    "Simple exit (no trailing stop)": ("Trailing stop", -1),
    "Add regime entry gate (rejected)": ("Regime entry gate", 1),
    "Add weekly filter (rejected)": ("Weekly filter", 1),
}

def evaluate_ticker(ticker, start = DATA_START, end = DATA_END, base_params = None):
    #top level so the worker processes in run_universe can call it
    data = DataHandler(ticker, start, end)
    wf = walk_forward(data, base_params = base_params, verbose = False)
    oos_start, oos_end = wf["equity_curve"].index[0], wf["equity_curve"].index[-1]

    report = performance_report(data, {"Strategy": wf})
    ablation = ablation_study(data, oos_start, oos_end, base_params = base_params)

    return {
        "ticker": ticker,
        "oos_start": oos_start,
        "oos_end": oos_end,
        "strategy": report["Strategy"].to_dict(),
        "buy_hold": report["Buy & Hold"].to_dict(),
        "mean_train_sharpe": wf["windows"]["train_sharpe"].mean(),
        "mean_test_sharpe": wf["windows"]["test_sharpe"].mean(),
        "ablation_sharpe_change": ablation.loc["sharpe_change"].drop("Full strategy").to_dict(),
    }

def run_universe(universe = UNIVERSE, start = DATA_START, end = DATA_END, max_workers = None, variants = None):
    #variants maps a name to param overrides; every (variant, ticker) pair is its own walk-forward
    variants = variants or {"Default": {}}
    tickers = [ticker for group in universe.values() for ticker in group]

    #download one at a time first, so the worker processes only ever read the cache
    for ticker in tickers:
        download_prices(ticker, start, end)

    #each walk-forward is independent, so they run in parallel (one is ~1.5 minutes on one core)
    jobs = [(name, ticker) for name in variants for ticker in tickers]
    max_workers = max_workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 2))
    results = {name: {} for name in variants}
    with ProcessPoolExecutor(max_workers = max_workers) as pool:
        futures = {pool.submit(evaluate_ticker, ticker, start, end, variants[name]): (name, ticker)
                   for name, ticker in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            name, ticker = futures[future]
            results[name][ticker] = future.result()
            print(f"[{done}/{len(jobs)}] {name}: {ticker} done", flush=True)

    return {name: [results[name][ticker] for ticker in tickers] for name in variants}

def universe_table(results, universe = UNIVERSE):
    group_of = {ticker: group for group, tickers in universe.items() for ticker in tickers}
    rows = []
    for result in results:
        strategy, buy_hold = result["strategy"], result["buy_hold"]
        rows.append({
            "ticker": result["ticker"],
            "group": group_of[result["ticker"]],
            "oos_from": result["oos_start"].year,
            "sharpe": strategy["sharpe"],
            "bh_sharpe": buy_hold["sharpe"],
            "sharpe_diff": strategy["sharpe"] - buy_hold["sharpe"],
            "cagr": strategy["cagr"],
            "bh_cagr": buy_hold["cagr"],
            "max_drawdown": strategy["max_drawdown"],
            "bh_max_drawdown": buy_hold["max_drawdown"],
            "calmar": strategy["calmar"],
            "bh_calmar": buy_hold["calmar"],
            "trades": strategy["trades"],
            "win_rate": strategy["win_rate"],
            "profit_factor": strategy["profit_factor"],
            "time_in_market": strategy["time_in_market"],
            "train_sharpe": result["mean_train_sharpe"],
            "test_sharpe": result["mean_test_sharpe"],
        })
    return pd.DataFrame(rows).set_index("ticker")

def format_universe_table(table):
    percent_columns = {"cagr", "bh_cagr", "max_drawdown", "bh_max_drawdown", "win_rate", "time_in_market"}
    formatted = table.copy()
    for column in formatted.columns:
        if column in percent_columns:
            formatted[column] = formatted[column].map(lambda value: f"{value * 100:.1f}%")
        elif column == "sharpe_diff":
            formatted[column] = formatted[column].map(lambda value: f"{value:+.2f}")
        elif column in ("trades", "oos_from"):
            formatted[column] = formatted[column].astype(int)
        elif column != "group":
            formatted[column] = formatted[column].map(lambda value: f"{value:.2f}")
    return formatted

def universe_headline(table):
    count = len(table)
    lines = [
        f"Sharpe beat buy & hold on {(table['sharpe_diff'] > 0).sum()} of {count} "
        f"(median {table['sharpe'].median():.2f} vs {table['bh_sharpe'].median():.2f})",
        f"Calmar beat buy & hold on {(table['calmar'] > table['bh_calmar']).sum()} of {count} "
        f"(median {table['calmar'].median():.2f} vs {table['bh_calmar'].median():.2f})",
        f"Max drawdown smaller on {(table['max_drawdown'] > table['bh_max_drawdown']).sum()} of {count} "
        f"(median {table['max_drawdown'].median():.1%} vs {table['bh_max_drawdown'].median():.1%})",
        f"Train to test decay: median per-window Sharpe {table['train_sharpe'].median():.2f} in training, "
        f"{table['test_sharpe'].median():.2f} out of sample",
    ]
    for group, members in table.groupby("group", sort=False):
        lines.append(f"  {group}: beat buy & hold Sharpe on {(members['sharpe_diff'] > 0).sum()} of {len(members)}, "
                     f"median Sharpe {members['sharpe'].median():.2f} vs {members['bh_sharpe'].median():.2f}")
    return lines

def universe_ablation(results):
    #contribution = sharpe with the component minus sharpe without it, so positive means it helps
    changes = pd.DataFrame({result["ticker"]: result["ablation_sharpe_change"] for result in results}).T
    contributions = pd.DataFrame({component: sign * changes[variant]
                                  for variant, (component, sign) in ABLATION_COMPONENTS.items()})
    summary = pd.DataFrame({
        "median": contributions.median(),
        "mean": contributions.mean(),
        "helps_on": (contributions > 0).sum(),
        "out_of": len(contributions),
    })
    return contributions, summary

def plot_universe(table, path = None):
    ordered = table.sort_values("sharpe_diff")
    positions = list(range(len(ordered)))

    with plt.rc_context({"font.family": ["Segoe UI", "DejaVu Sans"]}):
        fig, (ax_sharpe, ax_drawdown) = plt.subplots(1, 2, figsize=(12, 0.38 * len(ordered) + 1.8), sharey=True)
        fig.patch.set_facecolor(CHART_COLORS["surface"])

        for ax, strategy_column, benchmark_column, title in [
                (ax_sharpe, "sharpe", "bh_sharpe", "Sharpe ratio"),
                (ax_drawdown, "max_drawdown", "bh_max_drawdown", "Max drawdown")]:
            ax.hlines(positions, ordered[strategy_column], ordered[benchmark_column],
                      color=CHART_COLORS["axis"], linewidth=2, zorder=1)
            ax.scatter(ordered[benchmark_column], positions, s=60, zorder=2, color=CHART_COLORS["muted"],
                       edgecolors=CHART_COLORS["surface"], linewidths=1.5, label="Buy & hold")
            ax.scatter(ordered[strategy_column], positions, s=60, zorder=3, color=CHART_COLORS["strategy"],
                       edgecolors=CHART_COLORS["surface"], linewidths=1.5, label="Strategy")
            ax.set_title(title, loc="left", fontsize=11, color=CHART_COLORS["ink"])
            _style_axis(ax)

        ax_sharpe.axvline(0, color=CHART_COLORS["axis"], linewidth=0.8)
        ax_sharpe.set_yticks(positions)
        ax_sharpe.set_yticklabels(ordered.index, color=CHART_COLORS["ink_secondary"])
        ax_drawdown.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax_sharpe.legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=2, frameon=False, fontsize=9,
                         labelcolor=CHART_COLORS["ink_secondary"])

        fig.suptitle("Walk-forward out-of-sample results by ticker, sorted by Sharpe vs buy & hold",
                     x=0.01, ha="left", fontsize=13, color=CHART_COLORS["ink"])
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=150, facecolor=CHART_COLORS["surface"])
    return fig

def universe_report(universe = UNIVERSE, report_dir = REPORT_DIR):
    os.makedirs(report_dir, exist_ok=True)
    started = time.perf_counter()
    results = run_universe(universe)["Default"]
    print(f"Universe finished in {time.perf_counter() - started:.0f}s\n")

    table = universe_table(results, universe)
    print(format_universe_table(table).to_string())
    print()
    for line in universe_headline(table):
        print(line)

    contributions, summary = universe_ablation(results)
    print("\nAblation, Sharpe contribution of each component (with it minus without it, default params, positive = helps):")
    print(contributions.round(2).to_string())
    print(summary.round(2).to_string())

    table.to_csv(os.path.join(report_dir, "universe_results.csv"))
    contributions.to_csv(os.path.join(report_dir, "universe_ablation.csv"))
    plot_universe(table, path=os.path.join(report_dir, "universe_sharpe_drawdown.png"))
    print(f"\nCharts and tables saved to {report_dir}")

#experiments: each variant goes through the same walk-forward grid on the same universe, and the first one is the
#baseline the others are compared against. params are spelled out so the results don't depend on DEFAULT_PARAMS
#experiment 1: three ways of using the regime ("Stop width only" won and became the default)
REGIME_VARIANTS = {
    "Gate ADX 25": {"use_regime_filter": True, "adx_threshold": 25}, #entries only in trending regimes
    "Stop width only": {"use_regime_filter": False, "adx_threshold": 25}, #entries in any regime
    "Gate ADX 20": {"use_regime_filter": True, "adx_threshold": 20}, #same gate, "trending" switches on earlier
}

#experiment 2: does the regime's volatility half earn its place in the stop, or would one fixed k do as well?
#the fixed k is the midpoint of each grid pair (2, 2.75, 4), so both variants have the same average stop width
STOP_WIDTH_VARIANTS = {
    "Regime stop width": {"use_regime_filter": False, "regime_stop_width": True},
    "Fixed stop width": {"use_regime_filter": False, "regime_stop_width": False},
}

def variant_comparison(variant_results, universe = UNIVERSE):
    tables = {name: universe_table(results, universe) for name, results in variant_results.items()}
    baseline = next(iter(tables))

    sharpe = pd.DataFrame({name: table["sharpe"] for name, table in tables.items()})
    best = sharpe.idxmax(axis=1)
    sharpe["Buy & hold"] = tables[baseline]["bh_sharpe"]
    sharpe["best_variant"] = best

    summary = pd.DataFrame({name: {
        "median_sharpe": table["sharpe"].median(),
        "mean_sharpe": table["sharpe"].mean(),
        "median_diff_vs_baseline": (table["sharpe"] - tables[baseline]["sharpe"]).median(),
        "beats_baseline_on": (table["sharpe"] > tables[baseline]["sharpe"]).sum() if name != baseline else np.nan,
        "best_variant_on": (best == name).sum(),
        "beats_buy_hold_on": (table["sharpe_diff"] > 0).sum(),
        "median_calmar": table["calmar"].median(),
        "median_cagr": table["cagr"].median(),
        "median_max_drawdown": table["max_drawdown"].median(),
        "median_trades": table["trades"].median(),
        "median_time_in_market": table["time_in_market"].median(),
        "median_train_sharpe": table["train_sharpe"].median(),
        "median_test_sharpe": table["test_sharpe"].median(),
    } for name, table in tables.items()})

    return sharpe, summary

def format_variant_summary(summary, ticker_count):
    percent_rows = {"median_cagr", "median_max_drawdown", "median_time_in_market"}
    count_rows = {"beats_baseline_on", "best_variant_on", "beats_buy_hold_on"}

    def fmt(row, value):
        if pd.isna(value):
            return "-"
        if row in percent_rows:
            return f"{value * 100:.1f}%"
        if row in count_rows:
            return f"{int(value)} of {ticker_count}"
        if row == "median_diff_vs_baseline":
            return f"{value:+.2f}"
        return f"{value:.2f}"

    return pd.DataFrame({column: [fmt(row, summary.at[row, column]) for row in summary.index]
                         for column in summary.columns}, index=summary.index)

def variant_report(variants = REGIME_VARIANTS, universe = UNIVERSE, report_dir = REPORT_DIR, label = "regime"):
    os.makedirs(report_dir, exist_ok=True)
    started = time.perf_counter()
    variant_results = run_universe(universe, variants = variants)
    print(f"Variants finished in {time.perf_counter() - started:.0f}s\n")

    sharpe, summary = variant_comparison(variant_results, universe)
    print("Out-of-sample Sharpe by ticker:")
    print(sharpe.round(2).to_string())
    print()
    print(format_variant_summary(summary, len(sharpe)).to_string())

    sharpe.to_csv(os.path.join(report_dir, f"{label}_variants_sharpe.csv"))
    summary.to_csv(os.path.join(report_dir, f"{label}_variants_summary.csv"))
    print(f"\nTables saved to {report_dir}")


if __name__ == "__main__":
    #python Component.design.py                   -> full report for AAPL
    #python Component.design.py SPY               -> full report for another ticker
    #python Component.design.py --universe        -> walk-forward + ablation across the whole UNIVERSE
    #python Component.design.py --compare-regime  -> the REGIME_VARIANTS through the walk-forward on the UNIVERSE
    #python Component.design.py --compare-stop    -> the STOP_WIDTH_VARIANTS through the walk-forward on the UNIVERSE
    if len(sys.argv) > 1 and sys.argv[1] == "--universe":
        universe_report()
    elif len(sys.argv) > 1 and sys.argv[1] == "--compare-regime":
        variant_report(REGIME_VARIANTS, label = "regime")
    elif len(sys.argv) > 1 and sys.argv[1] == "--compare-stop":
        variant_report(STOP_WIDTH_VARIANTS, label = "stop_width")
    else:
        single_ticker_report(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
        plt.show()