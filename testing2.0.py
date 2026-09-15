import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

stock_data = yf.download("NVDA", "2021-01-02", "2026-02-28", interval = "1d")
stock_data.index = stock_data.index.strftime('%Y-%m-%d')
prices = stock_data["Close"].squeeze()
print(prices)

balance = 10000

nvda_price = prices.loc["2021-01-04"]

shares_num = balance/nvda_price

portfolio_value = {}
dates = stock_data.index

for i in range(len(prices)):
    portfolio_value[dates[i]] = shares_num * prices.loc[dates[i]]

fig, ax = plt.subplots()
ax.plot(prices.index, list(portfolio_value.values()))
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 120))
fig.autofmt_xdate()
plt.xlabel("Dates")
plt.ylabel("Portfolio Value in $")
plt.title("Nvidia Portfolio with Budget $10000")
plt.show()