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
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

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

def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=190, batch_size=30):
    """
    Trains the LSTM model and returns the training history.

    Parameters:
    model (Sequential): The LSTM model to be trained.
    X_train, y_train: Training data.
    X_val, y_val: Validation data.
    epochs (int): Number of training epochs.
    batch_size (int): Size of the batches used in training.

    Returns:
    History: The history object containing training and validation loss.
    """
    optimizer = optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    return history




def plot_loss(history, ticker, loss_label='Training loss', val_loss_label='Validation loss'):
    """
    Plots the training and validation loss.

    Parameters:
    history (History): The training history object containing loss information.
    ticker (str): The stock ticker symbol for title.
    loss_label (str): Label for the training loss curve.
    val_loss_label (str): Label for the validation loss curve.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.plot(epochs, loss, 'bo', label=loss_label)
    plt.plot(epochs, val_loss, 'b', label=val_loss_label)
    plt.title('Training and Validation Loss for ' + ticker)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



def plot_stock_predictions(y_test, predicted_stock_price, ticker, real_label='Real Stock Price', predicted_label='Predicted Stock Price'):
    """
    Plots the real and predicted stock prices.

    Parameters:
    y_test (array): The real stock prices.
    predicted_stock_price (array): The predicted stock prices.
    ticker (str): The stock ticker symbol for title.
    real_label (str): Label for the real stock price curve.
    predicted_label (str): Label for the predicted stock price curve.
    """
    plt.plot(y_test, color='red', label=real_label)
    plt.plot(predicted_stock_price, color='blue', label=predicted_label)
    plt.title('Stock Price Prediction for ' + ticker)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


def evaluate_model(y_test, predicted_stock_price):
    """
    Evaluates the LSTM model using various metrics.

    Parameters:
    y_test (array): The true stock prices.
    predicted_stock_price (array): The predicted stock prices by the model.

    Returns:
    dict: A dictionary containing evaluation metrics.
    """
    mse = mean_squared_error(y_test, predicted_stock_price)
    mape = mean_absolute_percentage_error(y_test, predicted_stock_price)

    evaluation_metrics = {
        "MSE": mse,
        "MAPE": mape
    }

    return evaluation_metrics






def lstm_hyperparameter_tuning(X_train, y_train, param_grid, n_iter=None):
    """
    Performs hyperparameter tuning for the LSTM model.

    Parameters:
    X_train (array): Training data features.
    y_train (array): Training data target.
    param_grid (dict): Dictionary with parameters names as keys and lists of parameter settings to try.
    n_iter (int, optional): Number of parameter settings that are sampled for RandomizedSearchCV.

    Returns:
    Best hyperparameters and corresponding model.
    """
    # Define a function to create a model (needed for KerasClassifier or KerasRegressor)
    def create_model(units=50, dropout_rate=0.2, optimizer='adam'):
        # Use the build_lstm_model function with dynamic parameters
        model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), units=units, dropout_rate=dropout_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    # Wrap the model with KerasRegressor or KerasClassifier
    model = KerasRegressor(build_fn=create_model, verbose=0)

    # Choose search method
    if n_iter:
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter, cv=3)
    else:
        search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)

    # Perform search
    search_result = search.fit(X_train, y_train)

    return search_result.best_params_, search_result.best_estimator_



def demonstrate_lstm_pipeline(ticker='DXCM', delay=5, lag=0, units=70, dropout_rate=0.2, epochs=190, batch_size=30):
    """
    Demonstrates the end-to-end pipeline of LSTM model for stock prediction.

    Parameters:
    ticker (str): The stock ticker symbol.
    delay (int): The delay parameter for preprocessing.
    lag (int): The lag parameter for preprocessing.
    units (int): The number of units in each LSTM layer.
    dropout_rate (float): The dropout rate for regularization.
    epochs (int): Number of training epochs.
    batch_size (int): Size of the batches used in training.

    Returns:
    dict: Evaluation metrics of the trained model.
    """
    # Data Preprocessing
    X, y, _, _, _ = preprocess_data(ticker, delay, lag)

    # Split data into train, test, and validation sets
    X_train, X_test, y_train, y_test, X_val, y_val = check_dimensions_and_split_data(X, y)


    # Model Building
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), units=units, dropout_rate=dropout_rate)

    # Model Training
    history = train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)

    # Plotting Loss
    plot_loss(history, ticker)

    # Making Predictions
    predicted_stock_price = model.predict(X_test)

    # Plotting Predictions
    plot_stock_predictions(y_test, predicted_stock_price, ticker)

    # Evaluating Model
    evaluation_metrics = evaluate_model(y_test, predicted_stock_price)

    return evaluation_metrics



def check_dimensions_and_split_data(X, y, test_size=0.2, val_size=0.2):
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


