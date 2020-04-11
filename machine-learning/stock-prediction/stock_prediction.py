from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import AveragePooling1D, GlobalAveragePooling1D, Flatten, TimeDistributed, InputLayer, Bidirectional, LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from sklearn import preprocessing, utils
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
from stockstats import StockDataFrame


import numpy as np
import pandas as pd
import random


def load_data(ticker, ticker_data, n_steps=50, shuffle=True, lookup_step=1,
                test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low'], 
                stat_columns=['macd'], target="adjclose"):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the training data, default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """
    if isinstance(ticker_data, pd.DataFrame):
        ticker = ticker_data


    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")


    feature_columns = np.concatenate((feature_columns, stat_columns), axis=None)

    sdf = StockDataFrame.retype(df.copy())

    for stat in stat_columns:
        df[stat] = sdf[stat]

    print(df)

    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()
    

    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns

    column_scaler = {}
    # scale the data (prices) from 0 to 1
    for column in feature_columns:
        scaler = preprocessing.MinMaxScaler()
        df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
        column_scaler[column] = scaler

    print(column_scaler)
    # add the MinMaxScaler instances to the result returned
    result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df[target].shift(-lookup_step)

    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    # drop NaNs
    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 59 (that is 50+10-1) length
    # this last_sequence will be used to predict in future dates that are not available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    # shift the last sequence by -1
    last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
    # add to result
    result['last_sequence'] = last_sequence

    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # reshape X to fit the neural network
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))

    print("Shapes")
    print (X.shape)

    # split the dataset
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle)

    # result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(
    #     X, y, test_size=test_size, shuffle=False)
    # if shuffle:
    #     result["X_train"], result["y_train"] = utils.shuffle(
    #         result["X_train"], result["y_train"])

    return result




def create_model(input_length, dropout=0.4,
                loss="mean_absolute_error", optimizer="rmsprop"):
    return create_model_lstm_3(input_length, dropout, loss, optimizer)


def create_model_conv(input_length, dropout, loss, optimizer):
    model = Sequential()
    model.add(Conv1D(filters=256, strides=1, kernel_size=2, use_bias=True,
                     activation='relu', input_shape=(None, input_length)))
    model.add(AveragePooling1D(pool_size=2, strides=1))

    # model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(dropout))

    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(1, activation="relu"))

    model.compile(loss=loss, metrics=[
                  "mean_absolute_error"], optimizer=optimizer)

    model.summary()
    return model



def create_model_conv_lstm(input_length, dropout, loss, optimizer):
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=1,
                     activation='relu', input_shape=(5, input_length)))
    model.add(Conv1D(filters=256, kernel_size=1, activation='relu'))
    # model.add(Dropout(dropout))
    model.add(MaxPooling1D(pool_size=2, padding="same"))

    model.add(Conv1D(filters=256, kernel_size=1, activation='relu'))
    model.add(Conv1D(filters=256, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding="same"))

    # model.add(GlobalAveragePooling1D(padding="same"))
    # model.add(Conv1D(filters=256, kernel_size=1, activation='relu'))
    # model.add(Conv1D(filters=256, kernel_size=1, activation='relu'))
    # model.add(MaxPooling1D(pool_size=2, padding="same"))

    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(192, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(1, activation="relu"))
    model.compile(loss=loss, metrics=[
                  "mean_absolute_error"], optimizer=optimizer)

    model.summary()
    return model

def create_model_mixed_bidirectional_lstm(input_length, dropout, loss, optimizer):
    model = Sequential()

    model.add(Bidirectional(LSTM(512, return_sequences=True),
                            input_shape=(None, input_length)))
    model.add(Dropout(dropout))
    model.add(LSTM(384, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(1, activation="relu"))
    model.compile(loss=loss, metrics=[
                  "mean_absolute_error"], optimizer=optimizer)

    return model


def create_model_lstm_3(input_length, dropout, loss, optimizer):
    model = Sequential()

    model.add(LSTM(256, return_sequences=True,
                   input_shape=(None, input_length)))
    model.add(Dropout(dropout))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=[
                  "mean_absolute_error"], optimizer=optimizer)

    return model


def create_model_bidirectional_lstm(input_length, dropout, loss, optimizer):
    model = Sequential()

    model.add(Bidirectional(LSTM(256, return_sequences=True),
                            input_shape=(None, input_length)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(192, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(dropout))

    model.add(Dense(1, activation="relu"))
    model.compile(loss=loss, metrics=[
                  "mean_absolute_error"], optimizer=optimizer)

    return model


def create_model_lstm_simplified(input_length, dropout, loss, optimizer):
    model = Sequential()

    model.add(LSTM(256, return_sequences=True, input_shape=(None, input_length)))
    model.add(Dropout(dropout))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)

    return model


def create_model_lstm_original(input_length, dropout=0.4,
                loss="mean_absolute_error", optimizer="rmsprop"):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            model.add(LSTM(units, return_sequences=True, input_shape=(None, input_length)))
        elif i == n_layers - 1:
            # last layer
            model.add(LSTM(units, return_sequences=False))
        else:
            # hidden layers
            model.add(LSTM(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))

    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)

    return model
