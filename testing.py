import pandas as pd
import yfinance as yf

raw_data = pd.DataFrame(yf.download("AAPL", "2025-01-01", "2025-01-20", interval="1d"))
raw_data.columns = raw_data.columns.get_level_values(0)
raw_data.index = [f"Day {i}" for i in range(1, len(raw_data) + 1)]

close_prices = pd.Series(raw_data["Close"])
print(raw_data.columns)



'''
date = {}
for i, price in enumerate(close_prices, start = 1):
    date[f"Day {i}"] = price


print(date)
'''
### Potentially reusable. Fully debugged
'''
    def moving_average_slope(self, window = 20, lookback = 20):
        dates = self.close.index
        moving_average = self.close.rolling(window=window).mean()
        slope = {}
        for i in range(len(dates)):
            if i < window -1 + lookback:
                slope[dates[i]] = None
            else:
                prev = moving_average.iloc[i - lookback]
                curr = moving_average.iloc[i]
                slope[dates[i]] = (curr - prev) / prev

        return pd.Series(slope)
'''