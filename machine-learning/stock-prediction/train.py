from model import create_model
from load_data import load_data
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.utils import class_weight
import os
import pandas as pd
import numpy as np
from parameters import *

# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")

if not os.path.isdir("logs"):
    os.mkdir("logs")

if not os.path.isdir("data"):
    os.mkdir("data")

# load the data
data = load_data(train_files, n_steps=N_STEPS, lookup_step=LOOKUP_STEP,
                 test_size_in_days=TEST_SIZE_IN_DAYS, feature_columns=FEATURE_COLUMNS,  
                 stat_columns=STAT_COLUMNS, target=TARGET)

# construct the model
model = create_model(N_STEPS, N_FEATURES,
                     loss=LOSS, optimizer=OPTIMIZER)

# some tensorflow callbacks
checkpointer = ModelCheckpoint(os.path.join("results", model_name), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=PATIENCE, verbose=1, mode='auto', baseline=None)

y_trainints = [y.argmax() for y in data["y_train"]]
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y_trainints),
                                                  y_trainints)
# class_weights = 1.0 - (data["y_train"].sum(axis=0) / float(data["y_train"].sum()))

print("Class weights: ", class_weights,
      ", sums: ", data["y_train"].sum(axis=0))
weights = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}
history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    class_weight=weights,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard, earlystopping],
                    verbose=2)

model.save(os.path.join("results", model_name) + ".h5")
