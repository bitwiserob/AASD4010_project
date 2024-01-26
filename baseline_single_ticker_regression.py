from single_preprocessing_function import single_preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt
import numpy as np

def single_ticker_baseline_regression(ticker='DXCM', delay=5, lag=0):
    #get pre processed data using our previous built function
    X, y, ticker, delay, lag = single_preprocessing(ticker=ticker, delay=delay, lag=lag)

    #Confirm selected ticker
    print("Executing baseline ANN for selected ticker :",ticker)
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

    #Simple ANN architechture
    model = Sequential()
    model.add(layers.Flatten(input_shape=(delay, X_train.shape[-1])))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(1))

    #Compile model and fit
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit(X_train, y_train, steps_per_epoch=150, epochs=20, validation_data=(X_val, y_val))

    #Plot training and validation loss
    print("\n")
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.title('Training and validation loss '+ticker)
    plt.legend()
    plt.show()

    #Make predictions and plot with correct values
    print("\n")
    predicted_stock_price = model.predict(X_test)
    plt.plot(y_test, color='red', label='Real Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
    plt.title('Stock Price Prediction ANN '+ticker)
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