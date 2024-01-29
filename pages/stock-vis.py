import dash
from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go
import pandas as pd


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
     Input('feature-selection', 'value')]
)
def update_graph(ticker, selected_features):
    stock = Stock(ticker)

    # Add selected features to the stock data
    if 'SMA' in selected_features:
        sma = stock.calculate_sma(20)  # Example window size
        stock.data['SMA'] = sma  # Ensure SMA is added to the DataFrame
    if 'EMA' in selected_features:
        ema = stock.calculate_ema(20)  # Example window size
        stock.data['EMA'] = ema  # Ensure EMA is added to the DataFrame

    df = stock.get_data()

    # Create the candlestick graph
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])

    # Add lines for each selected feature
    for feature in selected_features:
        if feature in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[feature], mode='lines', name=feature))

    fig.update_layout(title=f'Stock Data for {ticker}', xaxis_title='Date', yaxis_title='Price')
    return fig