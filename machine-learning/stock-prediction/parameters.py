import os
import time
import pandas as pd
import tensorflow as tf
import glob
from os import listdir
from os.path import isfile, join
from tensorflow.keras.layers import LSTM


# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
TARGET = "adjclose"
# STAT_COLUMNS = ["macd", "kdjk"]
STAT_COLUMNS = []
# date now
date_now = time.strftime("%Y-%m-%d")

### data
dataDir = "data"
test_ticker = "FB"
test_files = glob.glob(join(dataDir, test_ticker + '*.csv'))
train_files = glob.glob(join(dataDir, '*.csv'))
# train_files = test_files

# RELATIVIZE_VALUES = True # prices from one day divided by the day before.

# mean squared error loss
# LOSS = "mse"
LOSS = 'binary_crossentropy'

# OPTIMIZER = "adam"
OPTIMIZER = "rmsprop"
BATCH_SIZE = 256 # train x batches at once
EPOCHS = 600
PATIENCE = 35

# Window size or the sequence length
N_STEPS = 120
N_FEATURES = len(FEATURE_COLUMNS) + len(STAT_COLUMNS)
# Lookup step, 1 is the next day
LOOKUP_STEP = 30

# test ratio size, 0.2 is 20%
# TEST_SIZE = 0.2
TEST_SIZE_IN_DAYS = 30*12*3


# model name to save
model_name = f"{date_now}-seq-{N_STEPS}-features-{N_FEATURES}-lookupstep-{LOOKUP_STEP}-opt-{OPTIMIZER}-batch-{BATCH_SIZE}"
