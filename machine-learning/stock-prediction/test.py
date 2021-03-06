from model import create_model
from load_data import load_data
from parameters import *
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import sys


def inverse_transform_sequence_scaling(y_test, y_pred):
    for i in range (0, len(y_test)):
        scaler = data["scalers_test"][i]
        # y_pred[i] = scaler.inverse_transform(y_pred[i].reshape(-1, 1))[0][0]
        y_pred[i] = scaler.inverse_transform(y_pred[i].reshape(-1, 1))[0][0]
        y_test[i] = scaler.inverse_transform(y_test[i].reshape(-1, 1))[0][0]
    return y_test, y_pred


# def plot_graph(model, data):
#     y_test = data["y_test"]
#     X_test = data["X_test"]
#     y_pred = model.predict(X_test)

#     if 'last_sequence_scaler' in data:
#         y_test, y_pred = inverse_transform_sequence_scaling(y_test, y_pred)

#     y_test = np.squeeze(data["column_scaler"][TARGET].inverse_transform(np.expand_dims(y_test, axis=0)))
#     y_pred = np.squeeze(data["column_scaler"]
#                         [TARGET].inverse_transform(y_pred))        
#     plt.plot(y_test, c='b')
#     plt.plot(y_pred, c='r')
#     plt.xlabel("Days")
#     plt.ylabel("Price")
#     plt.legend(["Actual Price", "Predicted Price"])
#     plt.show()


def mae_pred_and_test(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)


def get_accuracy_buy_hold_sell(model, data, plot=False):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)

    m = tf.keras.metrics.CategoricalAccuracy()
    m.update_state(y_test, y_pred)

    if plot:
        # fig, ax = plt.subplots()
        fig, ax = plt.subplots(4, sharex='all',
                            gridspec_kw=dict(height_ratios=[4, 1,1,1]))

        category_test = np.array([tf.math.argmax(x).numpy() for x in y_test])
        category_pred = np.array([tf.math.argmax(x).numpy() for x in y_pred])

        matches = (category_test == category_pred).astype(int)
        # .shift(LOOKUP_STEP)
        price = data['df'][TARGET].values[-(len(category_test) +
                                            LOOKUP_STEP):-LOOKUP_STEP]
        ax[0].plot(price, 'k')
        
        ax[1].imshow(category_test[None, :], cmap=ListedColormap(["green", "yellow", "red"]), aspect='auto',
                     vmin=0.0, vmax=2.0)

        ax[2].imshow(category_pred[None, :], cmap=ListedColormap(["green", "yellow", "red"]), aspect='auto',
                    vmin=0.0, vmax=2.0)

        ax[3].imshow(matches[None, :], cmap=ListedColormap(["red", "green"]), aspect='auto',
                     vmin=0.0, vmax=2.0)

        plt.xlabel("Days")
        plt.legend(["Actual Price", "Predicted Price"])

        # fig.show()
        plt.subplots_adjust(hspace=0)
        plt.show()



    return m.result().numpy()


def get_accuracy_and_plot(model, data, plot=False):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)

    if 'last_sequence_scaler' in data:
        y_test, y_pred = inverse_transform_sequence_scaling(y_test, y_pred)

    y_test = np.squeeze(data["column_scaler"][TARGET].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"][TARGET].inverse_transform(y_pred))

    if plot:
        plt.plot(y_test, c='b')
        plt.plot(y_pred, c='r')
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend(["Actual Price", "Predicted Price"])
        plt.show()


    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))
    y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))

    return accuracy_score(y_test, y_pred)


def predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"]
    # last_sequence = data["last_sequence"][:N_STEPS]
    column_scaler = data["column_scaler"]
    
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)

    # get the price (by inverting the scaling)
    if 'last_sequence_scaler' in data:
        prediction = data["last_sequence_scaler"].inverse_transform(
            prediction[0][0].reshape(-1, 1))

    predicted_price = column_scaler[TARGET].inverse_transform(prediction)[0][0]
    return predicted_price


def predict_category(model, data):
    last_sequence = data["last_sequence"]

    last_sequence = np.expand_dims(last_sequence, axis=0)
    prediction = model.predict(last_sequence)

    labels = ['buy', 'hold', 'sell']
    return labels[tf.math.argmax(prediction[0]).numpy()]


# load the data
data = load_data(test_files, n_steps=N_STEPS, lookup_step=LOOKUP_STEP, test_size_in_days=TEST_SIZE_IN_DAYS,
                 feature_columns=FEATURE_COLUMNS, stat_columns=STAT_COLUMNS, target=TARGET, shuffle=False)

# construct the model
model = create_model(N_STEPS, N_FEATURES, loss=LOSS, optimizer=OPTIMIZER)
model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)

print(f"Counts: ", data["y_test"].sum(axis=0))
# evaluate the model
loss, acc = model.evaluate(data["X_test"], data["y_test"])
print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
# calculate the mean absolute error (inverse scaling)
# mae_inverted = data["column_scaler"][TARGET].inverse_transform(mae.reshape(1, -1))[0][0]

# mae_pred_test = mae_pred_and_test(model, data)

# # predict the future price
# future_price = predict(model, data)
# print(f"Fteuture price after {LOOKUP_STEP} days is {future_price:.2f}$")
# accuracy = get_accuracy_and_plot(model, data, len(sys.argv)>1)*100.0
accuracy = get_accuracy_buy_hold_sell(model, data, len(sys.argv) > 1)*100.0
print(f"Accuracy Score: {accuracy:.2f}%",)

prediction = predict_category(model, data)
print(f"Prediction in the next", LOOKUP_STEP, "days:", prediction)

# show_plot = sys.argv[1]
# if show_plot:
# plot_graph(model, data)
