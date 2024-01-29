from dash import Dash, dcc,Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
app = Dash(__name__, external_stylesheets=[dbc.themes.SIMPLEX])

text = dcc.Markdown(children="# hello world")
input01 = dbc.Input(value='text')
app.layout=dbc.Container([text])

app.layout = dbc.Container([text,input01])


@app.callback(
    Output(text, component_property='children'),
    Input(input01, component_property='value')
)

def update_title(user_input):
    return user_input




if __name__=='__main__':
    app.run_server(port=5000,debug=True)