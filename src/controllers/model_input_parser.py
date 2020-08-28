"""
Created by: Tapan Sharma
Date: 28/08/20
"""
import pandas as pd

from enum import Enum
from predictCO2.preprocessing import utils


class DataAnalysingModels(Enum):
    SOCIAL_POLICY_MODEL = 1,
    STRINGENCY_INDEX_MODEL = 2


class ParseModelInputs:
    def __init__(self, model):
        self.model_type = None
        self.stringency_idx = None
        self.school_closing_value = None
        self.workspace_closing_value = None
        self.public_events_value = None
        self.gathering_restrictions_value = None
        self.public_transport_value = None
        self.stay_home_value = None
        self.internal_movement_value = None
        self.international_travel_value = None

    def get_social_policy_data(self):
        incomplete_flagger = (self.school_closing_value >= 0) and (self.workspace_closing_value >= 0) and \
                             (self.public_events_value >= 0) and (self.gathering_restrictions_value >= 0) and \
                             (self.public_transport_value >= 0) and (self.stay_home_value >= 0) and \
                             (self.internal_movement_value >= 0) and (self.international_travel_value >= 0)
        if not incomplete_flagger:
            raise ValueError("Social policy values not set: {}".format(self.social_policy_values()))
        else:
            df = pd.DataFrame([[self.school_closing_score, self.workspace_closing_score, self.public_events_score,
                                self.gathering_restrictions_score, self.public_transport_score, self.stay_home_score,
                                self.internal_movement_score, self.international_travel_score]])
            df = pd.concat([df] * 30)  # Assuming 7 time steps.
            df = utils.data_sequence_generator(df, None, 7)
            return df

    @property
    def stringency_index(self):
        return self.stringency_idx

    @stringency_index.setter
    def stringency_index(self, value):
        self.stringency_idx = value

    @property
    def school_closing_score(self):
        return self.school_closing_value

    @school_closing_score.setter
    def school_closing_score(self, value):
        self.school_closing_value = value

    @property
    def workspace_closing_score(self):
        return self.workspace_closing_value

    @workspace_closing_score.setter
    def workspace_closing_score(self, value):
        self.workspace_closing_value = value

    @property
    def public_events_score(self):
        return self.public_events_value

    @public_events_score.setter
    def public_events_score(self, value):
        self.public_events_value = value

    @property
    def gathering_restrictions_score(self):
        return self.gathering_restrictions_value

    @gathering_restrictions_score.setter
    def gathering_restrictions_score(self, value):
        self.gathering_restrictions_value = value

    @property
    def public_transport_score(self):
        return self.public_transport_value

    @public_transport_score.setter
    def public_transport_score(self, value):
        self.public_transport_value = value

    @property
    def stay_home_score(self):
        return self.stay_home_value

    @stay_home_score.setter
    def stay_home_score(self, value):
        self.stay_home_value = value

    @property
    def internal_movement_score(self):
        return self.internal_movement_value

    @internal_movement_score.setter
    def internal_movement_score(self, value):
        self.internal_movement_value = value

    @property
    def international_travel_score(self):
        return self.international_travel_value

    @international_travel_score.setter
    def international_travel_score(self, value):
        self.international_travel_value = value

    def social_policy_values(self):
        return "School Closing: {}\nWorkspace Closing: {}\nPublic Events: {}\nGathering Restrictions: {}\n" \
               "Public Transport: {}\nStay at Home: {}\nInternal Movement: {}\nInternational Travel: {}".format(
                self.school_closing_score, self.workspace_closing_score, self.public_events_score,
                self.gathering_restrictions_score, self.public_transport_score, self.stay_home_score,
                self.internal_movement_score, self.international_travel_score)
