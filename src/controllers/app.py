# -*- coding: utf-8 -*-
# @Time    : 8/20/20 2:39 PM
# @Author  : Saptarshi
# @Email   : saptarshi.mitra@tum.de
# @File    : app.py
# @Project: group07

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
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
        # marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(1, 25)},
        value=5,
    ),
    html.Div(id='stringency_index_show')
], style={'columnCount': 2})#for two column view in HTML page

@app.callback(
    Output(component_id='stringency_index_show', component_property='children'),
    [Input(component_id='stringency_index', component_property='value')]
)
def update_stringency_index(input_value):
    return 'Stringency Index: {}'.format(input_value)

if __name__ == '__main__':
    app.run_server(debug=True)