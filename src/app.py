# -*- coding: utf-8 -*-
# @Time    : 8/20/20 2:39 PM
# @Author  : Saptarshi
# @Email   : saptarshi.mitra@tum.de
# @File    : app.py
# @Project: group07

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, meta_tags=[
    {
        'name': 'PredictCarbon COVID',
        'content': 'This is a group project completed for the course Applied Machine Intelligence at the Technical University of Munich'
    },
    {
        'http-equiv': 'X-UA-Compatible',
        'content': 'IE=edge'
    },
    {
      'name': 'viewport',
      'content': 'width=device-width, initial-scale=1.0'
    }
], title='PredictCarbon COVID', update_title='Calculating')

app.layout = html.Div([
    #html.Hr(style={'height':20,'border-width':0,'background-color':'#004BA0', 'margin-top':0, 'margin-left':0}),
    html.H1('Carbon Emissions Prediction by Policy Decisions'),
    html.Label('Dropdown'),
    dcc.Dropdown(
        options=[
            {'label': 'Italy', 'value': 'ITA'},
            {'label': 'South Korea', 'value': 'KOR'},
            {'label': 'Finland', 'value': 'FIN'},
            {'label': 'Brazil', 'value': 'BRA'}
        ],
        value='ITA'
    ),

    html.Label('Multi-Select Dropdown'),
    dcc.Dropdown(
        options=[
            {'label': 'Italy', 'value': 'ITA'},
            {'label': 'South Korea', 'value': 'KOR'},
            {'label': 'Finland', 'value': 'FIN'},
            {'label': 'Brazil', 'value': 'BRA'}
        ],
        value=['ITA', 'FIN'],
        multi=True
    ),

    html.Label('Radio Items'),
    dcc.RadioItems(
        options=[
            {'label': 'Italy', 'value': 'ITA'},
            {'label': 'South Korea', 'value': 'KOR'},
            {'label': 'Finland', 'value': 'FIN'},
            {'label': 'Brazil', 'value': 'BRA'}
        ],
        value='ITA'
    ),

    html.Label('Checkboxes'),
    dcc.Checklist(
        options=[
            {'label': 'Italy', 'value': 'ITA'},
            {'label': 'South Korea', 'value': 'KOR'},
            {'label': 'Finland', 'value': 'FIN'},
            {'label': 'Brazil', 'value': 'BRA'}
        ],
        value=['ITA', 'FIN']
    ),

    html.Label('Text Input'),
    dcc.Input(value='hi', type='text'),

    html.Label('Stringency Index'),
    dcc.Slider(
        id= 'stringency_index',
        min=0,
        max=100,
        marks={
            0: {'label': '0', 'style': {'color': '#77b0b1'}},
            25: {'label': '25'},
            50: {'label': '50'},
            75: {'label':'75'},
            100: {'label': '100', 'style': {'color': '#f50'}}
        },
        value=5,
    ),
    html.Div(id='stringency_index_show')
], style={'columnCount': 1})#for one column view in HTML page

@app.callback(
    Output(component_id='stringency_index_show', component_property='children'),
    [Input(component_id='stringency_index', component_property='value')]
)
def update_stringency_index(input_value):
    return 'Stringency Index: {}'.format(input_value)




if __name__ == '__main__':
    app.run_server(debug=True)