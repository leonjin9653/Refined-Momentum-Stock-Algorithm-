import yfinance as yf
import pandas as pd
ticker = yf.Ticker("AGI.to")

print(ticker.info.keys())


