# lstm_model.py
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras import layers, optimizers
from keras.layers import Dropout
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from single_preprocessing_function import single_preprocessing
import itertools

class LSTMModel:
    def __init__(self, ticker='DXCM', delay=5, lag=0, units=70, dropout_rate=0.2, epochs=190, batch_size=30):
        """
        Initializes the LSTMModel class with default parameters.

        Parameters:
        ticker (str): The stock ticker symbol.
        delay (int): The delay parameter for preprocessing.
        lag (int): The lag parameter for preprocessing.
        units (int): The number of units in each LSTM layer.
        dropout_rate (float): The dropout rate for regularization.
        epochs (int): Number of training epochs.
        batch_size (int): Size of the batches used in training.
        """
        self.ticker = ticker
        self.delay = delay
        self.lag = lag
        self.units = units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
    
    def preprocess_data(self):
        """
        Preprocesses the data for the LSTM model using the provided ticker, delay, and lag parameters.
        """
        X, y, _, _, _ = single_preprocessing(ticker=self.ticker, delay=self.delay, lag=self.lag)

        # Splitting data into train, test, and validation sets
        self.X_train, self.X_test, self.y_train, self.y_test, self.X_val, self.y_val = self.check_dimensions_and_split_data(X, y)

    def build_lstm_model(self, learning_rate=0.02, optimizer_choice='sgd' ):
        """
        Builds and returns an LSTM model based on the class parameters for units and dropout rate.
        """
        self.model = Sequential()
        self.model.add(layers.LSTM(units=self.units, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), activation="tanh"))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(layers.LSTM(units=self.units, return_sequences=False, activation="tanh"))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(layers.Dense(1))  # No activation function for regression
  
        if optimizer_choice == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'sgd':
            optimizer = optimizers.SGD(learning_rate=learning_rate)
        # Add other optimizers as needed

        self.model.compile(optimizer=optimizer, loss='mean_squared_error')



    def train_lstm_model(self, epochs=190, batch_size=30):
        """
        Trains the LSTM model.

        Parameters:
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        """
        if self.model is None:
            self.build_lstm_model()
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_val, self.y_val))
        
        return self.history
    
    def plot_loss(self):
        """
        Plots the training and validation loss.
        """
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(loss) + 1)

        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title(f'Training and Validation Loss for {self.ticker}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_stock_predictions(self):
        """
        Plots the real and predicted stock prices.
        """
        predicted_stock_price = self.model.predict(self.X_test)

        plt.plot(self.y_test, color='red', label='Real Stock Price')
        plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
        plt.title(f'Stock Price Prediction for {self.ticker}')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

    def evaluate_model(self):
        """
        Evaluates the LSTM model using mean squared error and mean absolute percentage error.
        """
        predicted_stock_price = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predicted_stock_price)
        mape = mean_absolute_percentage_error(self.y_test, predicted_stock_price)

        evaluation_metrics = {
            "MSE": mse,
            "MAPE": mape
        }

        return evaluation_metrics
    


    def check_dimensions_and_split_data(self, X, y, test_size=0.2, val_size=0.2):
        """
        Checks dimensions of X and y, adjusts if necessary, and splits the data into training, validation, and test sets.

        Parameters:
        X (array): The feature data.
        y (array): The target data.
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float): The proportion of the training dataset to include in the validation split.

        Returns:
        tuple: Split data (X_train, X_test, y_train, y_test, X_val, y_val).
        """
        # Dimensional check
        if X.shape[0] > y.shape[0]:
            X = X[:-1]
            print("Dimensional adjustment for X performed")
        else:
            print("Dimension check passed")

        # Confirm shapes of X and y
        print("Shape of X:", X.shape)
        print("Shape of y:", y.shape)

        assert X.shape[0] == y.shape[0], "The number of observations for X and y should be the same"
        print("2nd Dimension check passed")

        # Split data into test, validation, train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)

        return X_train, X_test, y_train, y_test, X_val, y_val


    def demonstrate_lstm_pipeline(self):
        """
        Demonstrates the end-to-end pipeline of the LSTM model for stock prediction.
        """
        self.preprocess_data()
        self.build_lstm_model()
        self.train_lstm_model()
        self.plot_loss()
        self.plot_stock_predictions()
        return self.evaluate_model()


    def manual_grid_search(self, param_grid, epochs, batch_size):
        results = []

        # Create all combinations of parameters
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for params in param_combinations:
            # Set the current parameters
            self.units, self.dropout_rate = params['units'], params['dropout_rate']
            learning_rate, optimizer_choice = params['learning_rate'], params['optimizer']

            # Build and train model
            self.build_lstm_model(learning_rate, optimizer_choice)
            self.train_lstm_model(epochs, batch_size)

            # Evaluate model using MSE and MAPE
            predictions = self.model.predict(self.X_val)
            mse_score = mean_squared_error(self.y_val, predictions)
            mape_score = mean_absolute_percentage_error(self.y_val, predictions)

            # Store results
            results.append({'params': params, 'mse': mse_score, 'mape': mape_score})

        return results