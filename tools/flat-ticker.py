import pandas as pd
import numpy as np
from datetime import datetime
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()


nasdaq100 = pd.read_csv('./data/company_info/nasdaq100.csv')


def flat_ticker(nasdaq100):
    df = pd.read_pickle("./data/data_pkls/nasdaq100.pkl")
    ret_df = pd.DataFrame()
    ret_df.columns = df.columns
    for i in nasdaq100['Symbol']:
        temp =  df[i]
        temp['Ticker'] = i
        pd.concat(ret_df, temp, axis=0)
        break

        

flat_ticker(nasdaq100)
