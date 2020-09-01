"""
Author: Sreetama Sarkar
Date: 8/31/2020
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from controllers.model_input_parser import DataAnalysingModels
from predictCO2.models.deep_learning_model import DeepLearningModel
from predictCO2.models.stringency_model import stringency_model
from predictCO2.preprocessing import utils
from predictCO2.preprocessing.generate_data import DataType, CountryPolicyCarbonData, PolicyCategory


class GenerateOutput:
    def __init__(self, pred_steps=30):
        self.pred_steps = pred_steps
        self.co2_data_avlbl = None
        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None
        self.model_type = None

    def get_data_avlbl(self, country, model_type):
        """
        Get already available CO2 data for given country (currently available till 11th June)
        :param country: One of the selected countries in UI
        :return: None
        """
        if model_type == DataAnalysingModels.STRINGENCY_INDEX_MODEL:
            countryPolicyCarbonData = CountryPolicyCarbonData('training_data.yaml', country, include_flags=False,
                                                              policy_category=PolicyCategory.STRINGENCY_INDEX,
                                                              normalize=0)
            data_type = DataType.PANDAS_DF
            self.co2_data_avlbl = countryPolicyCarbonData.get_labels(data_type)
        else:
            countryPolicyCarbonData = CountryPolicyCarbonData('training_data.yaml', country, include_flags=False,
                                                              policy_category=PolicyCategory.SOCIAL_INDICATORS,
                                                              normalize=0)
            self.train_features, self.train_labels, self.test_features, \
            self.test_labels = countryPolicyCarbonData.split_train_test(fill_nan=False)
            self.co2_data_avlbl = countryPolicyCarbonData.get_labels(DataType.PANDAS_DF)

    def get_dates_whole_duration(self, country, model_type):
        """
        Get dates starting from Janyary 1 till end of prediction
        :param country: One of the selected countries in UI
        :return: Dates for whole duration as a pandas.DateTimeIndex object
        """
        if self.co2_data_avlbl is None:
            self.get_data_avlbl(country, model_type)
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

    def get_co2_whole_duration_social_indicators(self, user_features):
        with open('cfg/cnn_config.json') as f:
            training_config = json.load(f)
        train_x, train_y = utils.generate_time_series_df(self.train_features, self.train_labels,
                                                         training_config['time_steps'])
        test_x, test_y = utils.generate_time_series_df(self.test_features, self.test_labels,
                                                         training_config['time_steps'])
        print("Training X: {}\nTraining Y: {}".format(train_x.shape, train_y.shape))
        tss = TimeSeriesSplit()
        _, n_features = train_x.shape
        model = DeepLearningModel(training_config, num_features=n_features, num_outputs=1)
        for train_idx, test_idx in tss.split(train_x):
            X, X_val = train_x.iloc[train_idx], train_x.iloc[test_idx]
            Y, Y_val = train_y.iloc[train_idx], train_y.iloc[test_idx]
            features, labels = utils.data_sequence_generator(X, Y, training_config['time_steps'])
            val_f, val_l = utils.data_sequence_generator(X_val, Y_val, training_config['time_steps'])
            model.train_with_validation_provided(features, labels, val_f, val_l)
        co2_pred = model.generate_future_prediction(user_features, test_x, test_y, self.pred_steps)
        co2_data_pred = np.array(co2_pred).reshape(-1, 1)
        co2_total_duration = np.concatenate((self.co2_data_avlbl.to_numpy().reshape(-1, 1), co2_data_pred))
        return co2_total_duration

    def get_dataframe_for_plotting(self, parsed_model_input, countries):
        """
        Returns final dataframe for selected countries for plotting
        :param str_index: selected stringency level in UI
        :param countries: Countries selected in UI
        :return: Dataframe for plotting
        """
        combined_df = pd.DataFrame()
        for country in countries:
            self.model_type = parsed_model_input.model
            dates = self.get_dates_whole_duration(country, parsed_model_input.model)
            if self.model_type == DataAnalysingModels.STRINGENCY_INDEX_MODEL:
                try:
                    co2_reductions = self.get_co2_whole_duration(parsed_model_input.get_stringency_data(), country)
                except Exception:
                    raise
            elif self.model_type == DataAnalysingModels.SOCIAL_POLICY_MODEL:
                try:
                    co2_reductions = self.get_co2_whole_duration_social_indicators(parsed_model_input.
                                                                                   get_social_policy_data())
                except Exception:
                    raise
            else:
                raise NotImplementedError("ONLY STRINGENCY AND SOCIAL POLICY DATA ANALYSIS SUPPORTED!!!!")
            if co2_reductions is None:
                raise ValueError("Failed to predict data for given settings!!!!")
            df = pd.DataFrame()
            df['date'] = dates
            df['co2'] = co2_reductions
            df['country'] = country
            combined_df = pd.concat((combined_df, df))
        return combined_df
