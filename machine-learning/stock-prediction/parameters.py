import os
import time
import pandas as pd
from tensorflow.keras.layers import LSTM


# Window size or the sequence length
N_STEPS = 90
# Lookup step, 1 is the next day
LOOKUP_STEP = 30

# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# date now
date_now = time.strftime("%Y-%m-%d")

### model parameters

# N_LAYERS = 2
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
# UNITS = 128
# 40% dropout
DROPOUT = 0.4

### training parameters

# mean squared error loss
LOSS = "mse"
OPTIMIZER = "adam"
# OPTIMIZER = "rmsprop"
BATCH_SIZE = 64
EPOCHS = 300

# stock market
ticker = "FB"
date_now = "2020-03-29"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")

ticker_data = pd.read_csv(ticker_data_filename) 

# model name to save
model_name = f"{date_now}_{ticker}-{LOSS}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}"