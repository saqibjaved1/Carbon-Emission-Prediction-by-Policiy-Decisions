"""
Created by: Tapan Sharma
Date: 28/08/20
"""
import dash
import plotly.express as px
from dash_extensions.enrich import State, Output, Input, Trigger

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
        Output('stringency-slider-section', 'style'),
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
            fig = px.line(df, x='date', y='co2', color='country')
            fig.update_layout(transition_duration=500)
            return "Processing Finished!", dcc.Graph(id='co2-graph', figure=fig), dcc.Store(id='trigger')
        else:
            return "Press submit when ready.", None, dcc.Store(id='trigger')

    @app.callback(Output("submit_policy_selection", "disabled"),
                  Trigger("submit_policy_selection", "n_clicks"),
                  Trigger("trigger", "data"))
    def disable_submit_until_callback_completes():
        context = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
        return context == 'submit_policy_selection'
