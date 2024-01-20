import pandas as pd
import numpy as np
from datetime import datetime
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime

yf.pdr_override()


nasdaq100 = pd.read_csv('./data/company_info/nasdaq100.csv')


def get_ticker_price(df):
    symbols = df['Symbol']
    symbols = list(symbols)
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)
    res = yf.download(symbols,group_by="ticker",period='max')
    res.to_pickle('./nasdaq100.pkl')



def get_ticker_price_flat(df):
    symbols = df['Symbol'].tolist()
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)
    
    # Download historical prices for all tickers in one go
    prices = yf.download(symbols, start=start, end=end)
    
    # Reshape the DataFrame to include a 'Ticker' column
    df_with_prices = pd.melt(prices['Adj Close'].reset_index(), id_vars=['Date'], 
                             var_name='Ticker', value_name='Adj Close')
    
    # Merge with the original DataFrame on 'Date'
    df_merged = pd.merge(df, df_with_prices, how='left', on='Date')
    
    # Save the merged DataFrame to a pickle file
    df_merged.to_pickle('./nasdaq100_with_prices.pkl')
    
    return df_merged

# Example usage
# Assuming 'your_df' is your DataFrame with a 'Symbol' column
# and 'yf' is the yfinance library
your_df_with_prices = get_ticker_price(nasdaq100)




get_ticker_price_flat(nasdaq100)
