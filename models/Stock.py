import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
import numpy as np

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
        model = ARIMA(self.data['Close'], order=order)
        self.arima_model = model.fit()
        return self.arima_model

    def fit_sarimax(self, order, seasonal_order):
        model = SARIMAX(self.data['Close'], order=order, seasonal_order=seasonal_order)
        self.sarimax_model = model.fit()
        return self.sarimax_model

    def get_forecast(self, model_type, steps=7):
            if model_type == 'ARIMA' and self.arima_model:
                forecast = self.arima_model.get_forecast(steps=steps)
            elif model_type == 'SARIMAX' and self.sarimax_model:
                forecast = self.sarimax_model.get_forecast(steps=steps)
            else:
                raise ValueError("Model not fitted or unknown model type.")

            # Ensure the data index is in datetime format
            last_date = self.data.index[-1]

            if isinstance(last_date, pd.Period):
                last_date = last_date.to_timestamp()

            # Generate future dates for the forecast
            # Adjusted to not use the 'closed' parameter
            future_dates = pd.date_range(start=last_date, periods=steps + 1, freq='D')[1:]
            
            forecast_df = pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast.predicted_mean.values,
                'yhat_lower': forecast.conf_int().iloc[:, 0].values,
                'yhat_upper': forecast.conf_int().iloc[:, 1].values
            })
            return forecast_df
    def plot_with_forecast(self, forecast_df, model_type):
        fig = go.Figure()

        # Plot historical data
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Close'], mode='lines', name='Close Price'))

        # Plot SMA and EMA if calculated
        for feature_name, feature_data in self.features.items():
            fig.add_trace(go.Scatter(x=self.data.index, y=feature_data, mode='lines', name=feature_name))

        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name=f'{model_type} Forecast'))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], fill='tonexty', mode='lines', line=dict(width=0), name=f'{model_type} Upper Bound'))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], fill='tonexty', mode='lines', line=dict(width=0), name=f'{model_type} Lower Bound'))

            # Update layout
        fig.update_layout(title=f'Stock Data and {model_type} Forecast for {self.ticker}', xaxis_title='Date', yaxis_title='Price')
        fig.show()


