import pandas as pd
import numpy as np
import yfinance as yf

class DataHandler:
    def __init__(self, ticker, start, end):
        raw_data = yf.download(ticker, start, end, interval="1d")

        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)
        if raw_data.empty:
            raise ValueError (f"No data returned for {ticker}.")

        self.close = raw_data["Close"]
        self.high = raw_data["High"]
        self.low = raw_data["Low"]
        self.volume = raw_data["Volume"]
        self.open = raw_data["Open"]

        self.current_index = 0
        self.current_date = self.close.index[0]
        self.final_date = self.close.index[-1]

    def advance(self):
        if self.current_date != self.final_date:
            self.current_index += 1
            self.current_date = self.close.index[self.current_index]
            return True
        return False
    def get_current_price(self):
        return self.close.loc[self.current_date], self.high.loc[self.current_date], self.low.loc[self.current_date], self.volume.loc[self.current_date], self.open.loc[self.current_date]
    def get_data_up_to_current(self):
        return self.close.iloc[:self.current_index + 1], self.high.iloc[:self.current_index + 1], self.low.iloc[:self.current_index + 1], self.volume.iloc[:self.current_index + 1], self.open.iloc[:self.current_index + 1]
    def reset(self):
        self.current_index = 0
        self.current_date = self.close.index[0]
        return self.current_date, self.current_index

class RegimeDetector:
    def __init__(self, raw_data):
        self.close = pd.Series(raw_data["Close"])
        self.high = pd.Series(raw_data["High"])
        self.low = pd.Series(raw_data["Low"])

    def trend_strength(self, period = 14):
        adx = {}
        pos_dm = {}
        neg_dm = {}
        dates = self.close.index
        true_range = {}
        
        for i in range(1, len(dates)):
            current_trading_range = self.high.values[i] - self.low.values[i]
            upward_gap = abs(self.high.values[i] - self.close.values[i - 1])
            downward_gap = abs(self.low.values[i] - self.close.values[i - 1])

            true_range[dates[i]] = max(current_trading_range, upward_gap, downward_gap)

            up_move = self.high.values[i] - self.high.values[i - 1]
            down_move = self.low.values[i - 1] - self.low.values[i]

            if up_move > down_move and up_move > 0:
                pos_dm[dates[i]] = up_move
            else:
                pos_dm[dates[i]] = 0 
            if down_move >up_move and down_move > 0:
                neg_dm[dates[i]] = down_move
            else:
                neg_dm[dates[i]] = 0

        smoothed_TR = {}
        smoothed_pos_dm = {}
        smoothed_neg_dm = {}
        for _ in range(0, period):
            smoothed_TR[dates[_]] = None
            smoothed_pos_dm[dates[_]] = None
            smoothed_neg_dm[dates[_]] = None

        total_TR = 0
        total_pdm = 0 
        total_ndm = 0
        for k in range(1, period + 1):
            total_TR += true_range[dates[k]]
            total_pdm += pos_dm[dates[k]]
            total_ndm += neg_dm[dates[k]]
        smoothed_TR[dates[period]] = total_TR / period
        smoothed_pos_dm[dates[period]] = total_pdm / period
        smoothed_neg_dm[dates[period]] = total_ndm / period
        
        for j in range(period + 1, len(dates)):
            smoothed_TR[dates[j]] = smoothed_TR[dates[j-1]] - (smoothed_TR[dates[j - 1]]/period) + true_range[dates[j]]
            smoothed_pos_dm[dates[j]] = smoothed_pos_dm[dates[j - 1]] - (smoothed_pos_dm[dates[j - 1]]/period) + pos_dm[dates[j]]
            smoothed_neg_dm[dates[j]] = smoothed_neg_dm[dates[j - 1]] - (smoothed_neg_dm[dates[j - 1]]/period) + neg_dm[dates[j]]

        pos_di = {}
        neg_di = {}

        for _ in range(0, period):
            pos_di[dates[_]] = None
            neg_di[dates[_]] = None
        for l in range(period, len(dates)):
            pos_di[dates[l]] = 100 * smoothed_pos_dm[dates[l]]/smoothed_TR[dates[l]]
            neg_di[dates[l]] = 100* smoothed_neg_dm[dates[l]]/smoothed_TR[dates[l]]

        d_index = {}
        for _ in range(0, period):
            d_index[dates[_]] = None
        for q in range(period, len(dates)):
            if pos_di[dates[q]] == 0 and neg_di[dates[q]] == 0:
                d_index[dates[q]] = 0 
            else:
                d_index[dates[q]] = 100 * abs(pos_di[dates[q]] - neg_di[dates[q]]) / (pos_di[dates[q]] + neg_di[dates[q]])

        for _ in range(0, 2 * period):
            adx[dates[_]] = None
        total_di = 0 
        for k in range(period, 2 * period):
            total_di += d_index[dates[k]]
        adx[dates[2 * period]] = total_di /period
        for t in range((2 * period) + 1, len(dates)):
            adx[dates[t]] = adx[dates[t - 1]] - (adx[dates[t-1]]/period) + d_index[dates[t]]

        return pd.Series(adx)

    def classify_trend(self, adx, strength_threshold = 25):
        dates = self.close.index
        trend = {}

        adx = adx.astype(object).replace(np.nan, None)

        for i in range(0, len(dates)):
            if adx[dates[i]] is None:
                trend[dates[i]] = None
            elif adx[dates[i]] >= strength_threshold:
                trend[dates[i]] = "trending"
            elif adx[dates[i]] < strength_threshold:
                trend[dates[i]] = "ranging"

        return pd.Series(trend)

    def rolling_volatility(self, window = 20):
        returns = self.close.pct_change()
        rolling_volatility = returns.rolling(window=window).std()

        return rolling_volatility

    def volatility_percentile(self, rolling_volatility, lookback = 100):
        volatility_percentiles = rolling_volatility.rolling(lookback).rank(pct=True)

        return volatility_percentiles

    def classify_volatility(self, volatility_percentiles, pct_threshold = 0.7):
        dates = self.close.index
        volatility_classification = {}

        volatility_percentiles = volatility_percentiles.astype(object).replace(np.nan, None)

        for i in range(0, len(dates)):
            if volatility_percentiles[dates[i]] is None:
                volatility_classification[dates[i]] = None
            elif volatility_percentiles[dates[i]] >= pct_threshold:
                volatility_classification[dates[i]] = "high_vol"
            elif volatility_percentiles[dates[i]] < pct_threshold:
                volatility_classification[dates[i]] = "low_vol"

        return pd.Series(volatility_classification)

    def classify_regime(self, trend, volatility_classification):
        dates = self.close.index
        regime = {}

        for i in range(len(dates)):
            if trend[dates[i]] is None and volatility_classification[dates[i]] is None:
                regime[dates[i]] = None
            elif trend[dates[i]] == "trending" and volatility_classification[dates[i]] == "high_vol":
                regime[dates[i]] = "trending_high_vol"
            elif trend[dates[i]] == "trending" and volatility_classification[dates[i]] == "low_vol":
                regime[dates[i]] = "trending_low_vol"
            elif trend[dates[i]] == "ranging" and volatility_classification[dates[i]] == "high_vol":
                regime[dates[i]] = "ranging_high_vol"
            elif trend[dates[i]] == "ranging" and volatility_classification[dates[i]] == "low_vol":
                regime[dates[i]] = "ranging_low_vol"
            else:
                regime[dates[i]] = None

        return pd.Series(regime)

class MomentumEntryIndicators:
    def __init__(self, raw_data):
        self.close = pd.Series(raw_data["Close"])
        self.high = pd.Series(raw_data["High"])
        self.low = pd.Series(raw_data["Low"])

    def atr(self, period=14):
        dates = self.close.index
        true_range = {}
        
        for i in range(1, len(dates)):
            current_trading_range = self.high.values[i] - self.low.values[i]
            upward_gap = abs(self.high.values[i] - self.close.values[i - 1])
            downward_gap = abs(self.low.values[i] - self.close.values[i - 1])

            true_range[dates[i]] = max(current_trading_range, upward_gap, downward_gap)

        avg_true_range = {}

        for _ in range(0, period):
            avg_true_range[dates[_]] = None

        total_TR = 0
        for k in range(1, period + 1):
            total_TR += true_range[dates[k]]

        avg_true_range[dates[period]] = total_TR / period

        for j in range(period + 1, len(dates)):
            avg_true_range[dates[j]] = avg_true_range[dates[j-1]] - (avg_true_range[dates[j - 1]]/period) + true_range[dates[j]]


        return pd.Series(avg_true_range)

    def donchian_high(self, period = 20):
        
        


            

#testing

stock = DataHandler("AAPL", "2025-01-01", "2026-01-20")
regime = RegimeDetector({"Close": stock.close, "High": stock.high, "Low": stock.low})
rolling_volatility = regime.rolling_volatility()
volatility_percentiles = regime.volatility_percentile(rolling_volatility)
print(regime.classify_volatility(volatility_percentiles))


 