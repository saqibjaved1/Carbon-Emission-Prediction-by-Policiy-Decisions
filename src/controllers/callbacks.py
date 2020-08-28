"""
Created by: Tapan Sharma
Date: 28/08/20
"""
import json
from dash.dependencies import Output, Input, State

from controllers.model_input_parser import ParseModelInputs, DataAnalysingModels
from predictCO2.models.deep_learning_model import DeepLearningModel


def register_callbacks(app):
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
        Output(component_id='output-mode-selector', component_property='children'),
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
        # 1. Parse Inputs in a required structure.
        dataframe = None
        if input_switcher_state:
            parse_model_input = ParseModelInputs(DataAnalysingModels.STRINGENCY_INDEX_MODEL)
            parse_model_input.stringency_idx = float(stringency_idx)
        else:
            parse_model_input = ParseModelInputs(DataAnalysingModels.SOCIAL_POLICY_MODEL)
            parse_model_input.school_closing_score = int(school_closing_score)
            parse_model_input.workspace_closing_score = int(workspace_closing_score)
            parse_model_input.public_events_score = int(public_events_score)
            parse_model_input.gathering_restrictions_score = int(gathering_restrictions_score)
            parse_model_input.public_transport_score = int(public_transport_score)
            parse_model_input.stay_home_score = int(stay_home_score)
            parse_model_input.internal_movement_score = int(internal_movement_score)
            parse_model_input.international_travel_score = int(international_travel_score)
            dataframe, _ = parse_model_input.get_social_policy_data()

        print(dataframe.shape)
        with open('cfg/cnn_config.json') as f:
            training_config = json.load(f)
        cnn = DeepLearningModel(training_config, num_features=8, num_outputs=1)
        cnn.load()
        y = cnn.model.predict(dataframe)
        print(y)
