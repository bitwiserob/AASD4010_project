import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


class Stock:
    def __init__(self, ticker, start_date='2010-01-01', end_date=None):
        """
        Initializes the Stock class with a ticker, start date, and end date.

        Parameters:
        ticker (str): The stock ticker symbol.
        start_date (str, optional): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str, optional): The end date for the data in 'YYYY-MM-DD' format.
        """

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.features = {}
        self.load_data()

    def load_data(self):
        """
        Loads stock data for the specified ticker and date range from a pickle file.
        """
        df = pd.read_pickle("./data/data_pkls/nasdaq100.pkl")
        stock_data = df.get(self.ticker, pd.DataFrame())

        if self.start_date is not None:
            stock_data = stock_data[stock_data.index >= self.start_date]
        if self.end_date is not None:
            stock_data = stock_data[stock_data.index <= self.end_date]

        self.data = stock_data

    def calculate_sma(self, window):
        """
        Calculates and stores the Simple Moving Average (SMA) for the given window.

        Parameters:
        window (int): The number of periods to calculate the SMA.

        Returns:
        pandas.Series: The calculated SMA values.
        """
        sma = self.data["Close"].rolling(window=window).mean()
        self.features[f"SMA_{window}"] = sma
        return sma

    def calculate_ema(self, window):
        """
        Calculates and stores the Exponential Moving Average (EMA) for the given window.

        Parameters:
        window (int): The number of periods to calculate the EMA.

        Returns:
        pandas.Series: The calculated EMA values.
        """
        ema = self.data["Close"].ewm(span=window, adjust=False).mean()
        self.features[f"EMA_{window}"] = ema
        return ema

    def get_data(self):
        """
        Returns the stock data as a pandas DataFrame.

        Returns:
        pandas.DataFrame: The stock data.
        """
        return self.data

    def plot_data(self):
        """
        Plots the stock's close price and any calculated features.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.data["Close"], label="Close Price")
        for feature_name, feature_data in self.features.items():
            plt.plot(feature_data, label=feature_name)
        plt.title(f"Stock Data for {self.ticker}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    def fit_arima(self, order):
        """
        Fits an ARIMA model to the stock's close price data.

        Parameters:
        order (tuple): The (p, d, q) order of the ARIMA model.

        Returns:
        ARIMAResults: The fitted ARIMA model.
        """

        model = ARIMA(self.data["Close"], order=order)
        self.arima_model = model.fit()
        return self.arima_model

    def fit_sarimax(self, order, seasonal_order):
        """
        Fits a SARIMAX model to the stock's close price data.

        Parameters:
        order (tuple): The (p, d, q) order of the model.
        seasonal_order (tuple): The (P, D, Q, s) seasonal order of the model.

        Returns:
        SARIMAXResults: The fitted SARIMAX model.
        """
        model = SARIMAX(self.data["Close"], order=order, seasonal_order=seasonal_order)
        self.sarimax_model = model.fit()
        return self.sarimax_model
