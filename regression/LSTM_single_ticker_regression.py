from single_preprocessing_function import single_preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras import optimizers
from keras.layers import Dropout
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt
import numpy as np


def preprocess_data(ticker='DXCM', delay=5, lag=0):
    """
    Preprocesses the data for the LSTM model.

    Parameters:
    ticker (str): The stock ticker symbol.
    delay (int): The delay parameter for preprocessing.
    lag (int): The lag parameter for preprocessing.

    Returns:
    tuple: Processed X, y data and other necessary information.
    """
    # Here, we'll insert the existing preprocessing logic from the original function
    # ...
    X, y, ticker, delay, lag = single_preprocessing(ticker=ticker, delay=delay, lag=lag)

    return X, y, ticker, delay, lag

def build_lstm_model(input_shape, units=70, dropout_rate=0.2):
    """
    Builds and returns an LSTM model.

    Parameters:
    input_shape (tuple): The shape of the input data.
    units (int): The number of units in each LSTM layer.
    dropout_rate (float): The dropout rate for regularization.

    Returns:
    Sequential: The constructed LSTM model.
    """
    lstm_model = Sequential()
    lstm_model.add(layers.LSTM(units=units, return_sequences=True, input_shape=input_shape, activation="tanh"))
    lstm_model.add(Dropout(dropout_rate))
    lstm_model.add(layers.LSTM(units=units, return_sequences=False, activation="tanh"))
    lstm_model.add(Dropout(dropout_rate))
    lstm_model.add(layers.Dense(1))  # No activation function for regression

    return lstm_model


def single_ticker_LSTM_regression(ticker='DXCM', delay=5, lag=0):
    #get pre processed data using our previous built function
    X, y, ticker, delay, lag = single_preprocessing(ticker=ticker, delay=delay, lag=lag)

    #Confirm selected ticker
    print("Executing RNN-LSTM for selected ticker :",ticker)
    print("\n")
    print("delay :",delay," lag :",lag)
    print("\n")

    #dimentional check
    if X.shape[0] > y.shape[0]:
        X = X[:-1]
        print("Dimentional adjustment for X performed")
    else:
        print("Dimension check passsed")

    #Confirm shapes of X and y
    print("\n")
    print("Shape of X :")
    print(X.shape)
    print("\n")
    print("Shape of y :")
    print(y.shape)
    print("\n")

    if X.shape[0] == y.shape[0]:
        print("2nd Dimension check passed")
    else:
        assert X.shape[0] == y.shape[0], "The number of observations for X and y should be the same"
    print("\n")
    #Split data into test, validation, train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    #RNN-LSTM architechture
    lstm_model = Sequential()
    lstm_model.add(layers.LSTM(units=70, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), activation="tanh" )) #add first layer for lstm
    lstm_model.add(Dropout(0.2))
    lstm_model.add(layers.LSTM(units=70, return_sequences=False, activation="tanh") ) #add second lstm layer
    lstm_model.add(Dropout(0.2))
    lstm_model.add(layers.Dense(1)) #don't add activation function because we are doing regression

    #Compile model and fit
    optimizer = optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False)
    lstm_model.compile(optimizer=optimizer, loss='mean_squared_error')
    lstm_history = lstm_model.fit(X_train, y_train, epochs=190, batch_size=30, validation_data=(X_val, y_val))

    #Plot training and validation loss
    loss = lstm_history.history['loss']
    val_loss = lstm_history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.title('Training and validation loss '+ticker)
    plt.legend()
    plt.show()

    #Make predictions and plot with correct values
    predicted_stock_price = lstm_model.predict(X_test)
    plt.plot(y_test, color='red', label='Real Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
    plt.title('Stock Price Prediction RNN '+ticker)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    #Evaluate model
    #Evaluate regression model using mse and map
    print("MSE for model: ")
    print(mean_squared_error(y_test, predicted_stock_price))
    print("\n")
    print("MAPE for model: ")
    print(mean_absolute_percentage_error(y_test, predicted_stock_price))