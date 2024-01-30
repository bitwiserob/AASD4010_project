from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go
import pandas as pd


import dash_bootstrap_components as dbc
import dash


from  models.Stock import Stock
app = Dash(__name__,external_stylesheets=[dbc.themes.DARKLY,'./assets/custom.css'],use_pages=True)

navbar = dbc.NavbarSimple(
    brand="Stock Dashboard",
    brand_href="#",
    color="primary",
    dark=True,
    children=[
        dbc.NavItem(dbc.NavLink("Stock Data", href="/")),  # Add link here
        dbc.NavItem(dbc.NavLink("Clustering", href="/clustering")),  # Add link here
        dbc.NavItem(dbc.NavLink("Sentiment", href="/Sentiment")),  # Add link here

        dbc.NavItem(dbc.NavLink("Models", href="/models")),  # Add link here

    ],
)

# Rest of your app layout
app.layout = html.Div([
    navbar,
    dbc.Container([
        
    ],fluid=True),dash.page_container
])







if __name__ == '__main__':
    app.run(debug=True)