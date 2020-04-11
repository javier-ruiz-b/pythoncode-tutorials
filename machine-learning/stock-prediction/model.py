from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import AveragePooling1D, GlobalAveragePooling1D, Flatten, TimeDistributed, InputLayer, Bidirectional, LSTM, Dense, Dropout, Conv1D, MaxPooling1D

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
