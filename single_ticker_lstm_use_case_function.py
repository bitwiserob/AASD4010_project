from single_preprocessing_function import single_preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras import optimizers
from keras.layers import Dropout

import matplotlib.pyplot as plt
import numpy as np
import keras

def single_ticker_lstm_use_case(ticker='DXCM', delay=5, lag=0, verbose=1):
    keras.utils.set_random_seed(97) #Set seed for reproducibility
    #get pre processed data using our previous built function
    X, y, ticker, delay, lag = single_preprocessing(ticker=ticker, delay=delay, lag=lag)

    #Confirm selected ticker
    if verbose == 1:
        print("Executing RNN-LSTM for selected ticker :",ticker)
        print("\n")
        print("delay :",delay," lag :",lag)
        print("\n")

    #dimentional check
    if X.shape[0] > y.shape[0]:
        X = X[:-1]
        if verbose == 1:
            print("Dimentional adjustment for X performed")
    else:
        if verbose == 1:
            print("Dimension check passsed")

    #Confirm shapes of X and y
    if verbose == 1:
        print("\n")
        print("Shape of X :")
        print(X.shape)
        print("\n")
        print("Shape of y :")
        print(y.shape)
        print("\n")

    if X.shape[0] == y.shape[0]:
        if verbose == 1:
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
    lstm_history = lstm_model.fit(X_train, y_train, epochs=190, batch_size=30, validation_data=(X_val, y_val), verbose=verbose)

    #Make predictions
    predicted_stock_price = lstm_model.predict(X_test)

    #Evaluate model
    #Evaluate regression model using mse and map
    print("MSE for model: ")
    print(mean_squared_error(y_test, predicted_stock_price))
    print("\n")
    print("MAPE for model: ")
    print(mean_absolute_percentage_error(y_test, predicted_stock_price))

    return lstm_model, ticker, delay