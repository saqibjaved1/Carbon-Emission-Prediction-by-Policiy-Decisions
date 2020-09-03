"""
Created by: Tapan Sharma
Date: 28/08/20
"""
import pandas as pd
import plotly.express as px
from dash_extensions.enrich import State, Output, Input

from controllers.model_input_parser import ParseModelInputs, DataAnalysingModels
from controllers.model_output_generator import GenerateOutput

def register_callbacks(app, dcc):
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
        if not value:
            return {'maxHeight': '250px', 'overflow': 'scroll', 'display': 'block'}
        else:
            return {'maxHeight': '250px', 'overflow': 'scroll', 'display': 'none'}

    @app.callback(
        Output('stringency-slider-container', 'style'),
        [Input('input-switch', 'value')])
    def update_output(value):
        if value:
            return {'display': 'block'}
        else:
            return {'display': 'none'}

    @app.callback(
        Output(component_id='outputs', component_property='children'),
        [Input('submit_policy_selection', 'n_clicks')],
        [State('input-switch', 'value'),
         State('stringency_index', 'value'),
         State('school-closing', 'value'),
         State('workplace-closing', 'value'),
         State('public-events', 'value'),
         State('gatherings', 'value'),
         State('public-transport', 'value'),
         State('stay-home', 'value'),
         State('internal-movement', 'value'),
         State('international-travel', 'value'),
         State('country-dropdown', 'value')]
    )
    def submit_button_controller(n_clicks, input_switcher_state, stringency_idx, school_closing_score,
                                 workspace_closing_score,
                                 public_events_score, gathering_restrictions_score, public_transport_score,
                                 stay_home_score,
                                 internal_movement_score, international_travel_score, countries):
        if n_clicks > 0:
            parse_model_input = ParseModelInputs()
            parse_model_input.countries = countries
            if input_switcher_state:  # when the toggle button is on right(stringency slider is on display)
                parse_model_input.model_type = DataAnalysingModels.STRINGENCY_INDEX_MODEL
                parse_model_input.stringency_idx = float(stringency_idx)
            else:
                parse_model_input.model_type = DataAnalysingModels.SOCIAL_POLICY_MODEL
                parse_model_input.school_closing_score = int(school_closing_score)
                parse_model_input.workspace_closing_score = int(workspace_closing_score)
                parse_model_input.public_events_score = int(public_events_score)
                parse_model_input.gathering_restrictions_score = int(gathering_restrictions_score)
                parse_model_input.public_transport_score = int(public_transport_score)
                parse_model_input.stay_home_score = int(stay_home_score)
                parse_model_input.internal_movement_score = int(internal_movement_score)
                parse_model_input.international_travel_score = int(international_travel_score)
            out = GenerateOutput()
            df = out.get_dataframe_for_plotting(parse_model_input, countries)
            # TODO: Add median values.
            fig = px.line(df, x='Date', y='MtCO2/day', color='Country')
            fig.update_layout(shapes=[dict(type= 'line',
                                            yref= 'paper', y0= 0, y1= 1,
                                            xref= 'x', x0= pd.to_datetime('2020-06-11'), x1= pd.to_datetime('2020-06-11'),
                                            line = dict(
                                                    # color="Red",
                                                    # width=4,
                                                    dash="dot"))],
                              transition_duration=500)
            return dcc.Graph(id='co2-graph', figure=fig), dcc.Store(id='trigger')
        else:
            return None, dcc.Store(id='trigger')

    # @app.callback(Output("submit_policy_selection", "disabled"),
    #               Trigger("submit_policy_selection", "n_clicks"),
    #               Trigger("trigger", "data"))
    # def disable_submit_until_callback_completes():
    #     context = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    #     return context == 'submit_policy_selection'

    # @app.callback(
    #     Output('submit_policy_selection', 'className'),
    #     [
    #         Input('co2-graph', 'figure')
    #     ],
    # )
    # def set_trend_enter_button_loading(figure_changed):
    #     return "button is-large is-primary is-outlined"
    #
    # @app.callback(
    #     Output('submit_policy_selection', 'className'),
    #     [
    #         Input('submit_policy_selection', 'n_clicks')
    #     ],
    # )
    # def set_trend_enter_button_loading(n_clicks):
    #     return "button is-large is-primary is-outlined is-loading"

    # @app.callback(Output("loading-output-1", "children"), [Input("loading-input-1", "value")])
    # def input_triggers_spinner(value):
    #     time.sleep(1)
    #     return value