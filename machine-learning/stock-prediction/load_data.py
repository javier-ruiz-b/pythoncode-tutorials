from yahoo_fin import stock_info as si
from collections import deque
from stockstats import StockDataFrame
from sklearn import preprocessing, utils
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import random

def interpolate_missing_values(df, feature_columns):
    for column in feature_columns:
        df[column] = df[column].replace([np.inf, -np.inf, 0], np.nan)
        df[column] = df[column].interpolate()
    return df

def relativize_df(df, feature_columns):
    new_cols = ["date"] + feature_columns
    result = pd.DataFrame(columns=new_cols)
    result["date"] = df.values[1:,0]
    for column in feature_columns:
        col_data = df[column]
        col_data_shifted = col_data.shift(1)
        col_data_relativized = col_data / col_data_shifted
        # print(col_data_relativized[1:])
        result[column] = col_data_relativized.values[1:]
    
    # print(result)
    return result


def load_data(csv_files, relativize=False, n_steps=50, shuffle=True, lookup_step=1,
              test_size_in_days=40, feature_columns=['adjclose', 'volume', 'open', 'high', 'low'],
              stat_columns=['macd'], target="adjclose"):

    last_date = ""
    last_date_file = ""
    results = {}
    for file in csv_files:
        df = pd.read_csv(file)
        # if relativize:
        #     df = relativize_df(df, feature_columns)

        df = interpolate_missing_values(df, feature_columns)

        current_last_date = df.values[-1][0]
        if last_date == "":
            last_date = current_last_date
            last_date_file = file
        elif last_date != current_last_date:
            raise ValueError("Expecting same last date on data: ",
                             file, " last date is ", current_last_date  , " where ",
                             last_date_file, " date is ", last_date)
            
        results[file] = load_data_single(
            df, n_steps, shuffle, lookup_step, test_size_in_days, feature_columns, stat_columns, target)

    if len(csv_files) == 1:
        return results[csv_files[0]]
    
    result = {}
    for file in csv_files:
        for key in ["X_train", "X_test", "y_train", "y_test"]:
            if key in result:
                result[key] = np.append(result[key], results[file][key], axis=0)
            else:
                result[key] = results[file][key]


    train_samples = len(result["X_train"])
    test_samples = len(result["X_test"])
    
    test_percent = test_samples*100.0 / float(test_samples + train_samples)

    print(f"Train samples: {train_samples}. Test samples: {test_samples}. TestPercent: {test_percent:.2f}")
    print("Train shape: ", result["X_train"].shape)
    
    return result
    


def load_data_single(df, n_steps=50, shuffle=True, lookup_step=1,
              test_size_in_days=40, feature_columns=['adjclose', 'volume', 'open', 'high', 'low'],
              stat_columns=['macd'], target="adjclose", scale_sequences=True):
    """
    Loads data from dir, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the training data, default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """

    feature_columns = np.concatenate((feature_columns, stat_columns), axis=None)

    sdf = StockDataFrame.retype(df.copy())

    for stat in stat_columns:
        df[stat] = sdf[stat]

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
        df[column] = scaler.fit_transform(
            np.expand_dims(df[column].values, axis=1))
        column_scaler[column] = scaler

    # print(column_scaler)
    # add the MinMaxScaler instances to the result returned
    result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df[target].shift(-lookup_step)

    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    # last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # get last_sequence for prediction
    last_sequence = np.array(df[feature_columns].tail(n_steps))

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
    # last_sequence = list(sequences) + list(last_sequence)
    # # shift the last sequence by -1
    # last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())


    if scale_sequences:
        scaler = preprocessing.MinMaxScaler(feature_range=(0.3, 0.7))
        scaler.fit(last_sequence.flatten().reshape(-1, 1))
        last_sequence = scaler.transform(last_sequence)
        result['last_sequence_scaler'] = scaler

    # add to result
    result['last_sequence'] = last_sequence

    # construct the X's and y's
    X, y, scalers = [], [], []
    for seq, target in sequence_data:
        if scale_sequences:
            scaler = preprocessing.MinMaxScaler(feature_range=(0.3, 0.7))
            scaler.fit(seq.flatten().reshape(-1, 1))
            seq = scaler.transform(seq)
            target = scaler.transform(target.reshape(-1, 1))[0][0]
            scalers.append(scaler)

        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # reshape X to fit the neural network

    # X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))

    samples = len(y)
    test_size = test_size_in_days / float(samples)
    print(f"test_size: {test_size:.2f}")

    ### Split the dataset 
    ## This method isn't right. Mixes values from the same time periods
    # result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(
    #     X, y, test_size=test_size, shuffle=shuffle)

    ## Splits train and test data and then shuffle
    result["X_train"], result["X_test"], result["y_train"], result["y_test"], result["scalers_train"], result["scalers_test"] = train_test_split(
        X, y, scalers, test_size=test_size, shuffle=False)

    if shuffle:
        result["X_train"], result["y_train"] = utils.shuffle(
            result["X_train"], result["y_train"])

    return result
