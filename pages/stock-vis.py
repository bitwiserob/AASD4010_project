import dash
from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px


import dash_bootstrap_components as dbc



from  models.Stock import Stock
dash.register_page(__name__,path='/')
tickers = ['AAPL', 'GOOG', 'MSFT']  # Replace with your list of tickers
features = ['SMA', 'EMA']  # Add other features as needed


layout = html.Div([
    dbc.Container([
        dbc.Row(
            dbc.Col(html.H1('Stock Data Visualization', className='text-center mb-4'), width=8, lg={"size": 8, "offset": 2}),
            justify='center'
        ),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(tickers, tickers[0], id='ticker-selection'),
                html.Label('Select Features:', className='mt-3'),
                dcc.Checklist(
                    features,
                    [],
                    id='feature-selection',
                    inline=True
                ),
                html.Label('Cluster:', className='h5 mt-3'),
                html.Br(),
                html.Label('LABEL', className='mt-3'),
                html.Br(),
                html.Label('Recent Sentiment:', className='h5 mt-3'),
                html.Br(),
                html.Label('Bullish', className='mt-3'),
                html.Br(),
                dcc.Dropdown(['ARIMA', 'SARIMAX'], 'ARIMA', id='model-selection'),  # Dropdown for model selection

                dbc.Button('Run forecast',id='forecast-button', n_clicks=0),

                
            ], width=3, lg={"size": 3},class_name='info-details'),
            dbc.Col([
                dcc.Graph(id='stock-graph')
            ],width=8, lg={"size": 9})
        ],
         
        justify='center', ),

    ], fluid=True,className='max-width-container')
])
@callback(
    Output('stock-graph', 'figure'),
    [Input('ticker-selection', 'value'),
     Input('feature-selection', 'value'),
     Input('model-selection', 'value'),  # Input for model selection
     Input('forecast-button', 'n_clicks')]
)
def update_graph(ticker, selected_features, selected_model, n_clicks):
    stock = Stock(ticker)

    # Add selected features to the stock data
    if 'SMA' in selected_features:
        sma = stock.calculate_sma(20)  # Example window size
        stock.data['SMA'] = sma  # Ensure SMA is added to the DataFrame
    if 'EMA' in selected_features:
        ema = stock.calculate_ema(20)  # Example window size
        stock.data['EMA'] = ema  # Ensure EMA is added to the DataFrame

    df = stock.get_data()



    fig = px.line(df, x=df.index, y=df['Close'], title='Time Series with Range Slider and Selectors')

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1d", step="day", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )


    # Add lines for each selected feature
    for feature in selected_features:
        if feature in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[feature], mode='lines', name=feature))
        if n_clicks > 0:
            if selected_model == 'ARIMA':
                # Fit and forecast using ARIMA (example: order (1, 1, 1))
                model = stock.fit_arima(order=(1, 1, 1))
            elif selected_model == 'SARIMAX':
                # Fit and forecast using SARIMAX (example: order and seasonal_order)
                model = stock.fit_sarimax(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            
        forecast = model.forecast(steps=7)  # Forecasting for a week
        future_dates = pd.date_range(start=df.index[-1], periods=8, closed='right')
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Forecast'))

    # ... [rest of the update_gr
    fig.update_layout(title=f'Stock Data for {ticker}', xaxis_title='Date', yaxis_title='Price')
    return fig