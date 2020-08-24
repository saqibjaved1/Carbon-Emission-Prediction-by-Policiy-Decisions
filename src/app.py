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

app.layout = html.Div(
    children=[
        html.Div(className='appui',
            children=[
                html.Div(className='toparea',
                         children=[
                            html.H1('Carbon Emissions Prediction by Policy Decisions'),
                         ]
                ),
                html.Div(className='inputarea',
                        children=[
                            html.Label('Select the countries for analysis:', style={'margin-left': 20, 'color':'white'}),
                                dcc.Dropdown(className='dropdown',id='country-dropdown',
                                    options=[
                                        {'label': 'Italy', 'value': 'ITA'},
                                        {'label': 'South Korea', 'value': 'KOR'},
                                        {'label': 'Finland', 'value': 'FIN'},
                                        {'label': 'Brazil', 'value': 'BRA'}
                                    ],
                                    value=['ITA', 'FIN'],
                                    multi=True
                                ),
                            html.Div(className='indicators-scroll', style={'margin-left': 20, "maxHeight": "200px", "overflow": "scroll"},
                                children=[
                                    html.Label('Social Indicators:', style={'margin-left': 20, 'color':'white'}),
                                    html.Label('School Closing:', style={'margin-left': 20, 'color':'white'}),
                                    dcc.Checklist(className='checkboxes',id='school-closing',
                                        options=[
                                            {'label': 'No measures ', 'value': 'l1'},
                                            {'label': 'Recommend closing', 'value': 'l2'},
                                            {'label': 'Require closing (on some levels)', 'value': 'l3'},
                                            {'label': 'Require closing (on all levels)', 'value': 'l4'}
                                        ],
                                        value=['l1']
                                    ),
                                    html.Label('Workplace Closing:', style={'margin-left': 20, 'color':'white'}),
                                    dcc.Checklist(className='checkboxes', id='workplace-closing',
                                        options=[
                                            {'label': 'No measures ', 'value': 'l1'},
                                            {'label': 'Recommend closing', 'value': 'l2'},
                                            {'label': 'Require closing (on some levels)', 'value': 'l3'},
                                            {'label': 'Require closing (on all levels)', 'value': 'l4'}
                                        ],
                                        value=['l1']
                                    ),

                                ]
                            ),



                            # dcc.RadioItems(
                            #     options=[
                            #         {'label': 'Italy', 'value': 'ITA'},
                            #         {'label': 'South Korea', 'value': 'KOR'},
                            #         {'label': 'Finland', 'value': 'FIN'},
                            #         {'label': 'Brazil', 'value': 'BRA'}
                            #     ],
                            #     value='ITA'
                            # ),


                            # html.Label('Text Input'),
                            # dcc.Input(value='hi', type='text'),

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
                            html.Div(className='slider', id='stringency_index_show'),
                            html.Button('Submit', id='button')
                        ], style={'columnCount': 2}#for two column view in HTML page
                )
            ]
        )
    ]
)
#, style={'columnCount': 1}
    # html.Label('Dropdown'),
    # dcc.Dropdown(
    #     options=[
    #         {'label': 'Italy', 'value': 'ITA'},
    #         {'label': 'South Korea', 'value': 'KOR'},
    #         {'label': 'Finland', 'value': 'FIN'},
    #         {'label': 'Brazil', 'value': 'BRA'}
    #     ],
    #     value='ITA'
    # ),



@app.callback(
    Output(component_id='stringency_index_show', component_property='children'),
    [Input(component_id='stringency_index', component_property='value')]
)
def update_stringency_index(input_value):
    return 'Stringency Index: {}'.format(input_value)




if __name__ == '__main__':
    app.run_server(debug=True)