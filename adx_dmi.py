import pandas as pd
import numpy as np
import yfinance as yf
import ta

class amx_indicators():
    def __init__(self, ticker, start, end):
        raw_data = yf.download(ticker, start=start, end=end, interval= "1d")

        high = raw_data["High"]
        low = raw_data["Low"]
        close = raw_data["Close"]

        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        self.high = high
        self.low = low
        self.close = close

    def adx(self):
        self.adx_series = ta.trend.adx(high=self.high, low=self.low, close=self.close, window=14)
        return self.adx_series



### Testing ###

stock = amx_indicators("AAPL", "2020-01-01", "2021-01-01")
print(stock.adx())

