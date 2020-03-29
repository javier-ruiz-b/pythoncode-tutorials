from stock_prediction import create_model, load_data
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import os
import pandas as pd
from parameters import *


# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")

if not os.path.isdir("logs"):
    os.mkdir("logs")

if not os.path.isdir("data"):
    os.mkdir("data")

# load the data
data = load_data(ticker, ticker_data, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)

# construct the model
model = create_model(N_STEPS, loss=LOSS, dropout=DROPOUT, optimizer=OPTIMIZER)

# some tensorflow callbacks
checkpointer = ModelCheckpoint(os.path.join("results", model_name), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
earlystopping = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=PATIENCE, verbose=1, mode='auto', baseline=None)


history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard, earlystopping],
                    shuffle=False,
                    verbose=0)

model.save(os.path.join("results", model_name) + ".h5")
