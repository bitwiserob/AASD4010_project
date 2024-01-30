import dash
from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from models.Stock import Stock

dash.register_page(__name__, path='/')
tickers = ['AAPL', 'GOOG', 'MSFT']  # Replace with your list of tickers
lines = ['Close','Open','High', 'Low']  # Add other features as needed

layout = html.Div([
    dbc.Container([
        dbc.Row(
            dbc.Col(html.H1('Stock Data Visualization', className='text-center mb-4'), width=8, lg={"size": 8, "offset": 2}),
            justify='center'
        ),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(tickers, tickers[0], id='ticker-selection'),
                                dcc.Checklist(
                    lines,
                    ['Close'],  # Default checked values
                    id='selected_lines',
                    inline=True,
                    inputStyle={"margin-right": "5px", "margin-left": "10px"}  # Adding padding
                ),
            ], width=3, lg={"size": 3}, class_name='info-details'),
            dbc.Col([
                dcc.Graph(id='stock-graph')
            ], width=8, lg={"size": 9})
        ], justify='center'),
    ], fluid=True, className='max-width-container')
])

@callback(
    Output('stock-graph', 'figure'),
    [Input('ticker-selection', 'value'),
     Input('selected_lines', 'value')]
)
def update_graph(ticker,selected_lines):
    stock = Stock(ticker)
    df = stock.get_data()

    fig = go.Figure()

    for feature in selected_lines:
        if feature in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[feature], mode='lines', name=feature))

    # Update layout and range selectors
    fig.update_layout(title=f'Stock Data for {ticker}', xaxis_title='Date', yaxis_title='Price')
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

    return fig
