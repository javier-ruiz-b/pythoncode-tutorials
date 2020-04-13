from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import AveragePooling1D, GlobalAveragePooling1D, Flatten, TimeDistributed, InputLayer, Bidirectional, LSTM, Dense, Dropout, Conv1D, MaxPooling1D

def create_model(n_steps, n_features, dropout=0.4,
                loss="mean_absolute_error", optimizer="rmsprop"):
    model = create_model_lstm_3(n_steps, n_features, dropout, loss, optimizer)
    model.summary()
    return model


def create_model_conv(n_steps, n_features, dropout, loss, optimizer):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                     input_shape=(n_steps, n_features)))
    # model.add(Conv1D(filters=256, kernel_size=1))
    # model.add(Conv1D(filters=48, kernel_size=2, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(AveragePooling1D(pool_size=2))

    # model.add(Flatten())
    # model.add(Flatten())
    #p = {2, 3, 5}

    # model.add(Conv1D(filters=256, kernel_size=1, activation='relu'))
    # model.add(AveragePooling1D(pool_size=2, strides=1))

    # model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(dropout))

    # model.add(LSTM(48, return_sequences=True))
    # model.add(Dropout(dropout))

    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(1, activation="relu"))

    model.compile(loss=loss, metrics=[
                  "mean_absolute_error"], optimizer=optimizer)

    return model


def create_model_conv_lstm(n_steps, n_features, dropout, loss, optimizer):
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=1,
                     activation='relu', input_shape=(n_steps, n_features)))
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
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=[
                  "mean_absolute_error"], optimizer=optimizer)

    return model


def create_model_mixed_bidirectional_lstm(n_steps, n_features, dropout, loss, optimizer):
    model = Sequential()

    model.add(Bidirectional(LSTM(512, return_sequences=True),
                            input_shape=(n_steps, n_features)))
    model.add(Dropout(dropout))
    model.add(LSTM(384, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=[
                  "mean_absolute_error"], optimizer=optimizer)

    return model


def create_model_lstm_3(n_steps, n_features, dropout, loss, optimizer):
    model = Sequential()

    model.add(LSTM(64, return_sequences=True,
                   input_shape=(n_steps, n_features)))
    model.add(Dropout(dropout))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(dropout))
    # model.add(LSTM(256, return_sequences=True))
    # model.add(Dropout(dropout))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(32))
    model.add(Dense(1, activation="relu"))
    model.compile(loss=loss, metrics=[
                  "mean_absolute_error"], optimizer=optimizer)

    return model


def create_model_bidirectional_lstm(n_steps, n_features, dropout, loss, optimizer):
    model = Sequential()

    model.add(Bidirectional(LSTM(256, return_sequences=True),
                            input_shape=(n_steps, n_features)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(192, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(dropout))

    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=[
                  "mean_absolute_error"], optimizer=optimizer)

    return model


def create_model_lstm_simplified(n_steps, n_features, dropout, loss, optimizer):
    model = Sequential()

    model.add(LSTM(256, return_sequences=True, input_shape=(None, n_steps)))
    model.add(Dropout(dropout))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)

    return model


def create_model_lstm_original(n_steps, n_features, dropout=0.4,
                loss="mean_absolute_error", optimizer="rmsprop"):
    model = Sequential()
    n_layers = 3
    units = 256
    for i in range(n_layers):
        if i == 0:
            # first layer
            model.add(LSTM(units, return_sequences=True,
                           input_shape=(None, n_steps)))
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
