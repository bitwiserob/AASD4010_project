import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
def single_preprocessing(ticker='SPLK', drop_col='Close', delay=15, lag=0):
    df = pd.read_pickle("./data/data_pkls/nasdaq100.pkl") #read data
    #Filter dataframe by this specific ticker
    df = df[ticker]
    #Drop missing values since there is no good way to estimate stock values that are not tracked
    df.dropna(inplace=True)
    #Drop Close column since it Adj Close column is calcualted using this information
    #so it would be allowing the model to cheat by already knowing the answers
    df.drop(columns={drop_col}, inplace=True)

    if drop_col == 'Close':
        organize_col = 'Adj Close'
    else:
        organize_col = 'Close'
    #Now just organize the columns 
    df = df.loc[:, ['Open', 'High', 'Low', 'Volume', organize_col]]
    #Now let's scale our data
    #First let's split data into X and Y
    X = df.loc[:, ['Open', 'High', 'Low', 'Volume']]
    y = df['Adj Close']
    scaler = MinMaxScaler() #create scaler object
    scaled_data = scaler.fit_transform(X) #fit transform data
    #Split data into samples and reshape X
    samples = [ scaled_data[i:i+delay] for i in range(0, (y.shape[0]), delay)]
    if len(samples[-1] < delay):
        samples = np.array(samples[:-1])
    else:
        samples = np.array(samples)
    #Reshape y target also
    new_y = [ y[i+delay+lag] for i in range(0, (y.shape[0]-(delay+lag)), delay)]
    new_y = np.array(new_y)
    return samples, new_y, ticker, delay, lag