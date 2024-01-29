import dash
from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go
import pandas as pd


import dash_bootstrap_components as dbc



from  models.Stock import Stock
dash.register_page(__name__,path='/clustering')


# Rest of your app layout
layout = html.Div([

    dbc.Container([
        
    ],fluid=True),dash.page_container
])

