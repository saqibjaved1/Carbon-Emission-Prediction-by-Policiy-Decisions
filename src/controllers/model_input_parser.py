"""
Created by: Tapan Sharma
Date: 28/08/20
"""
from enum import Enum

import pandas as pd


class DataAnalysingModels(Enum):
    SOCIAL_POLICY_MODEL = 1,
    STRINGENCY_INDEX_MODEL = 2


class ParseModelInputs:
    def __init__(self):
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
        self.countries_list = None
        self.incomplete_flagger_social = None
        self.incomplete_flagger_stringency = None

    def get_social_policy_data(self):
        self.incomplete_flagger_social = (self.school_closing_value >= 0) and (self.workspace_closing_value >= 0) and \
                             (self.public_events_value >= 0) and (self.gathering_restrictions_value >= 0) and \
                             (self.public_transport_value >= 0) and (self.stay_home_value >= 0) and \
                             (self.internal_movement_value >= 0) and (self.international_travel_value >= 0)
        if not self.incomplete_flagger_social:
            raise ValueError("Social policy values not set: {}".format(self.social_policy_values()))
        else:
            df = pd.DataFrame().from_records({"20200612": [self.school_closing_value, self.workspace_closing_value,
                                                           self.public_events_value, self.gathering_restrictions_value,
                                                           self.public_transport_value, self.stay_home_value,
                                                           self.internal_movement_value,
                                                           self.international_travel_value]})
            return df.T

    def get_stringency_data(self):
        self.incomplete_flagger_stringency = self.stringency_index is not None
        if not self.incomplete_flagger_stringency:
            raise ValueError("Stringency Value not set: {}".format(self.stringency_values()))
        else:
            return self.stringency_index

    @property
    def model(self):
        return self.model_type

    @model.setter
    def model(self, value):
        self.model_type = value

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

    @property
    def countries(self):
        return self.countries_list

    @countries.setter
    def countries(self, value):
        self.countries_list = value

    def social_policy_values(self):
        return "School Closing: {}\nWorkspace Closing: {}\nPublic Events: {}\nGathering Restrictions: {}\n" \
               "Public Transport: {}\nStay at Home: {}\nInternal Movement: {}\nInternational Travel: {}".format(
                self.school_closing_score, self.workspace_closing_score, self.public_events_score,
                self.gathering_restrictions_score, self.public_transport_score, self.stay_home_score,
                self.internal_movement_score, self.international_travel_score)

    def stringency_values(self):
        return "Stringency Value: {}".format(self.stringency_index)