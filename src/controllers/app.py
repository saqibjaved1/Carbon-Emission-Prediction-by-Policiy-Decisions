# -*- coding: utf-8 -*-
# @Time    : 8/20/20 2:39 PM
# @Author  : Saptarshi
# @Email   : saptarshi.mitra@tum.de
# @File    : app.py.py
# @Project: group07

import dash
import dash_core_components as dcc
import dash_html_components as html

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
], style={'columnCount': 2})

if __name__ == '__main__':
    app.run_server(debug=True)