import os
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM


# Window size or the sequence length
N_STEPS = 50
# Lookup step, 1 is the next day
LOOKUP_STEP = 5

# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
TARGET = "adjclose"
STAT_COLUMNS = ["macd", "kdjk"]
# STAT_COLUMNS = []
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
# LOSS = tf.keras.losses.BinaryCrossentropy(from_logits=True)
OPTIMIZER = "adam"
# OPTIMIZER = "rmsprop"
BATCH_SIZE = 256 # train x batches at once
EPOCHS = 600
PATIENCE = 30

# stock market
ticker = "AAPL"
date_now = "2020-01-08"
# ticker = "FB"
# date_now = "2020-03-29"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")

ticker_data = pd.read_csv(ticker_data_filename) 

# model name to save
model_name = f"{date_now}_{ticker}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}"
