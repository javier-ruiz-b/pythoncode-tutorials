from yahoo_fin import stock_info as si
import pandas as pd
import sys
import os
import time

date_now = time.strftime("%Y-%m-%d")
ticker = sys.argv[1]
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")

df = si.get_data(ticker)
df.to_csv (ticker_data_filename, index = False, header=True)
