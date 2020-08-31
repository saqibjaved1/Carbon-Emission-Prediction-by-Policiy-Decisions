"""
Author: Sreetama Sarkar
Date: 8/31/2020
"""

import numpy as np
import pandas as pd
from predictCO2.preprocessing.generate_data import DataType, CountryPolicyCarbonData, PolicyCategory
from predictCO2.models.stringency_model import stringency_model

class GenerateOutput:
    def __init__(self, pred_steps = 30):
        self.pred_steps = pred_steps

    def get_co2_data_avlbl(self, country):
        """
        Get already available CO2 data for given country (currently available till 11th June)
        :param country: One of the selected countries in UI
        :return: None
        """
        countryPolicyCarbonData = CountryPolicyCarbonData('training_data.yaml', country, include_flags=False,
                                                          policy_category=PolicyCategory.STRINGENCY_INDEX,
                                                          normalize=0)
        data_type = DataType.PANDAS_DF
        self.co2_data_avlbl = countryPolicyCarbonData.get_labels(data_type)


    def get_dates_whole_duration(self, country):
        """
        Get dates starting from Janyary 1 till end of prediction
        :param country: One of the selected countries in UI
        :return: Dates for whole duration as a pandas.DateTimeIndex object
        """
        self.get_co2_data_avlbl(country)
        avlbl_dates = pd.to_datetime(self.co2_data_avlbl.columns)
        next_dates = pd.date_range(avlbl_dates[-1], periods=self.pred_steps + 1)
        dates = avlbl_dates.union(next_dates)
        return dates


    def get_co2_whole_duration(self, str_index, country):
        """
        Combine CO2 predictions with available CO2 values
        :param str_index: selected stringency level in UI
        :param country: One of the selected countries in UI
        :return: CO2 values for whole duration as a numpy array
        """
        str_model = stringency_model(stringency=str_index, country=country, pred_steps=self.pred_steps)
        co2_pred = str_model.generate_future_prediction()
        co2_data_pred = np.array(co2_pred).reshape(-1, 1)
        co2_total_duration = np.concatenate((self.co2_data_avlbl.to_numpy().reshape(-1, 1), co2_data_pred))
        return co2_total_duration


    def get_dataframe_for_plotting(self, str_index, countries):
        """
        Returns final dataframe for selected countries for plotting
        :param str_index: selected stringency level in UI
        :param countries: Countries selected in UI
        :return: Dataframe for plotting
        """
        combined_df = pd.DataFrame()
        for country in countries:
            dates = self.get_dates_whole_duration(country)
            co2_reductions = self.get_co2_whole_duration(str_index, country)
            df = pd.DataFrame()
            df['date'] = dates
            df['co2'] = co2_reductions
            df['country'] = country
            combined_df = pd.concat((combined_df, df))
        return combined_df