from yahoo_fin import stock_info as si
import pandas as pd
import sys
import os
import time

def download(ticker):
    date_now = time.strftime("%Y-%m-%d")
    ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")

    df = si.get_data(ticker=ticker, interval="1d")
    df.to_csv (ticker_data_filename, index=True, header=True)

if len(sys.argv) > 1:
    ticker = sys.argv[1]
    download(ticker)