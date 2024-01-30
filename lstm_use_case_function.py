#Now let's make a function that predicts the following 5 days of the value of the stock for a selected ticker
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
from single_preprocessing_use_case_function import single_preprocessing_use_case
from single_ticker_lstm_use_case_function import single_ticker_lstm_use_case

def lstm_use_case(ticker='DXCM', delay=5, lag_list=[0, 1, 2, 3, 4], verbose=0):
    keras.utils.set_random_seed(97) #Set seed for reproducibility
    #Sanity check
    if len(lag_list) > delay:
        assert "The number of days to predict ahead cannot be greater than delay, check lag_list and delay"
    #Init empty list to store predictions
    predictions_list = []
    #Train model and make predictions in a loop
    for lag in lag_list:
        #Build a model to predict the price of DXCM stock on the next day 
        lstm, _, _ = single_ticker_lstm_use_case(ticker=ticker, delay=delay, lag=lag, verbose=0)
        #Get observation
        observation, _, _, _, _ = single_preprocessing_use_case(ticker=ticker, delay=delay, lag=lag)
        #Make prediction
        prediction = lstm.predict(observation)
        #Add to list of predictions
        predictions_list.append(prediction[0][0])

    #Plot predictions for the next days
    lag_list = np.array(lag_list)+1
    plt.figure()
    plt.plot(lag_list, predictions_list, 'r', label='Predictions')
    plt.xlabel('Days ahead')
    plt.title('Predictions for the next '+str(len(lag_list))+' days')
    plt.legend()
    plt.show()