# -*- coding: utf-8 -*-
# @Time    : 8/20/20 2:39 PM
# @Author  : Saptarshi
# @Email   : saptarshi.mitra@tum.de
# @File    : app.py
# @Project: group07

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np

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

country_names = np.array(pd.ExcelFile('dataset/features/Modified_Stringency_Data.xlsx').sheet_names)
country_list = [{'label': i, 'value': i[:3]} for i in country_names]

app.layout = html.Div(
    children=[
        html.Div(className='app-ui',
            children=[
                html.Div(className='top-area',
                         children=[
                            html.H1('Carbon Emissions Prediction by Policy Decisions'),
                         ]
                ),
                html.Div(className='input-area',
                        children=[
                            html.Label('Select the countries for analysis:', style={'margin-left': 20, 'color':'white'}),
                                dcc.Dropdown(className='dropdown',id='country-dropdown',
                                    options=country_list,
                                    value=['Ita', 'Fin'],
                                    multi=True
                                ),

                            daq.ToggleSwitch(
                                id='input-switch',
                                value=False,
                                size=70
                            ),
                            html.Div(className='indicators-scroll', id='social-indicators-scroll',style={"maxHeight": "200px", "overflow": "scroll",'display': 'block'},#block this div by toggle button
                                children=[
                                    html.Label('Social Indicators:', style={'margin-left': 20, 'color':'black'}),
                                    html.Label('1. School Closing:', style={'margin-left': 20, 'color':'black'}),
                                    dcc.RadioItems(className='checkboxes',id='school-closing',
                                        options=[
                                            {'label': 'No measures ', 'value': 'l1'},
                                            {'label': 'Recommend closing', 'value': 'l2'},
                                            {'label': 'Require closing (on some levels)', 'value': 'l3'},
                                            {'label': 'Require closing (on all levels)', 'value': 'l4'}
                                        ],
                                        value='l1'
                                    ),
                                    html.Label('2. Workplace Closing:', style={'margin-left': 20, 'color':'black'}),
                                    dcc.RadioItems(className='checkboxes', id='workplace-closing',
                                        options=[
                                            {'label': 'No measures ', 'value': 'l1'},
                                            {'label': 'Recommend closing', 'value': 'l2'},
                                            {'label': 'Require closing (for some sectors)', 'value': 'l3'},
                                            {'label': 'Require closing (for all sectors)', 'value': 'l4'}
                                        ],
                                        value='l1'
                                    ),
                                    html.Label('3. Cancel public events:', style={'margin-left': 20, 'color':'black'}),
                                    dcc.RadioItems(className='checkboxes', id='public-events',
                                        options=[
                                            {'label': 'No measures ', 'value': 'l1'},
                                            {'label': 'Recommend cancelling', 'value': 'l2'},
                                            {'label': 'Require cancelling', 'value': 'l3'}
                                        ],
                                        value='l1'
                                    ),
                                    html.Label('4. Restrictions on gatherings:', style={'margin-left': 20, 'color':'black'}),
                                    dcc.RadioItems(className='checkboxes', id='gatherings',
                                        options=[
                                            {'label': 'No restrictions', 'value': 'l1'},
                                            {'label': 'Restrictions on very large gatherings (the limit is above 1000 people)', 'value': 'l2'},
                                            {'label': 'Restrictions on gatherings between 101-1000 people', 'value': 'l3'},
                                            {'label': 'Restrictions on gatherings between 11-100 people', 'value': 'l4'},
                                            {'label': 'Restrictions on gatherings of 10 people or less', 'value': 'l5'}
                                        ],
                                        value='l1'
                                    ),
                                    html.Label('5. Close public transport:', style={'margin-left': 20, 'color':'black'}),
                                    dcc.RadioItems(className='checkboxes', id='public-transport',
                                        options=[
                                            {'label': 'No measures ', 'value': 'l1'},
                                            {'label': 'Recommend closing (or significantly reduce volume/route/means)', 'value': 'l2'},
                                            {'label': 'Require closing (or prohibit most citizens from using it)', 'value': 'l3'}
                                        ],
                                        value='l1'
                                    ),
                                    html.Label('6. Stay at home requirements:', style={'margin-left': 20, 'color':'black'}),
                                    dcc.RadioItems(className='checkboxes', id='stay-home',
                                        options=[
                                            {'label': 'No measures ', 'value': 'l1'},
                                            {'label': 'Recommend not leaving house', 'value': 'l2'},
                                            {'label': 'Require not leaving house with exceptions for daily exercise, grocery shopping, ...', 'value': 'l3'},
                                            {'label': 'Require not leaving house with minimal exceptions', 'value': 'l4'}
                                        ],
                                        value='l1'
                                    ),
                                    html.Label('7. Restrictions on internal movement:', style={'margin-left': 20, 'color':'black'}),
                                    dcc.RadioItems(className='checkboxes', id='internal-movement',
                                        options=[
                                            {'label': 'No measures ', 'value': 'l1'},
                                            {'label': 'Recommend not to travel between regions/cities', 'value': 'l2'},
                                            {'label': 'Internal movement restrictions in place', 'value': 'l3'},
                                        ],
                                        value='l1'
                                    ),
                                    html.Label('8. International travel controls:', style={'margin-left': 20, 'color':'black'}),
                                    dcc.RadioItems(className='checkboxes', id='international-travel',
                                        options=[
                                            {'label': 'No measures ', 'value': 'l1'},
                                            {'label': 'Screening arrivals', 'value': 'l2'},
                                            {'label': 'Quarantine arrivals from some or all regions', 'value': 'l3'},
                                            {'label': 'Ban arrivals from some regions', 'value': 'l4'},
                                            {'label': 'Ban on all regions or total border closure', 'value': 'l5'}
                                        ],
                                        value='l1'
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

                            html.Div(className='slider', id='stringency-slider-section',style={'display': 'block'},
                                    children=[
                                        html.Label('Stringency Index', style={'margin-left':20,'color':'white'}),
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
                                        html.Div(className='slider-data', id='stringency_index_show', style={'margin-left':20,'color':'white'}),
                                    ]
                            ),
                            html.Img(src='assets/map.png', height=400,width=700)
                        ], style={'columnCount': 2}#for two column view in HTML page
                ),
                html.Div(className='mid-area',
                         children=[
                            html.Button('Submit', id='button', className='submit-button')
                         ]
                ),
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


@app.callback(
    Output('social-indicators-scroll', 'style'),
    [Input('input-switch', 'value')])
def update_output(value):
    if value==False:
        return {'maxHeight': '250px', 'overflow': 'scroll','display': 'block'}
    else:
        return {'maxHeight': '250px', 'overflow': 'scroll','display': 'none'}


@app.callback(
    Output('stringency-slider-section', 'style'),
    [Input('input-switch', 'value')])
def update_output(value):
    if value==True:
        return {'display': 'block'}
    else:
        return {'display': 'none'}



if __name__ == '__main__':
    app.run_server(debug=True)