"""
Created by: Tapan Sharma
Date: 20/6/2020
"""

import abc
import pandas as pd
import predictCO2.preprocessing.utils as utils

from enum import Enum


class DataType(Enum):
    """
    Enum specifying type in which training data is to be made available.
    """
    DICT = 1  # For type dictionary
    PANDAS_DF = 2  # For type pandas data frame


class TrainDataInterface(object):
    """
    Interface representing training data. Defined below are abstract methods that are implemented by the respective
    sub-classes for training.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_features(self, in_form: DataType):
        """
        Return features as specified by the type
        :param in_form: either Dictionary or Pandas Data frame
        """
        pass

    @abc.abstractmethod
    def get_labels(self, in_form: DataType):
        """
        Return labels as specified by the type
        :param: in_form: either Dictionary or Pandas Data frame
        """
        pass

    @abc.abstractmethod
    def get_augmented_data(self, in_form: DataType):
        """
        Return augmented data combining features with labels
        :param in_form: either Dictionary or Pandas Data frame
        """
        pass

    @abc.abstractmethod
    def save_data_frame_to_csv(self, location: str):
        """
        Save data frame as csv
        :param location: Location on local file system
        """
        pass


class CountryPolicyCarbonData(TrainDataInterface):
    def __init__(self, policy_csv, carbon_csv):
        """
        Class to keep Policy and Carbon data by country.
        Uses OXFORD OxCGRT (https://github.com/OxCGRT/covid-policy-tracker) as features/policy
        Uses Global Carbon Project (https://www.icos-cp.eu/gcp-covid19) as labels/carbon
        :param policy_csv: Path to policy/features csv file
        :param carbon_csv: Path to carbon/label csv file
        """
        self.policy_df = pd.read_csv(policy_csv)
        self.carbon_df = pd.read_csv(carbon_csv)
        self.country_name = self.carbon_df.iloc[1, :]['REGION_NAME']
        self.combined_data_dict = {}
        self.feature_dict = {}
        self.label_dict = {}
        self.num_features = 0

    def get_features(self, data_type):
        """
        Return features as specified by argument
        :param data_type DataType either DICT or PANDAS_DF
        :rtype: python dictionary or pandas data frame
        """
        if not self.feature_dict:
            for index, row in self.policy_df.iterrows():
                if row['CountryName'] != self.country_name:
                    continue
                else:
                    date = row['Date']
                    policy = CountryPolicyCarbonData.policy_values_as_list(row)
                    self.feature_dict[str(date)] = policy
            self.num_features = len(next(iter(self.feature_dict.values())))
        if data_type == DataType.DICT:
            return self.feature_dict
        if data_type == DataType.PANDAS_DF:
            countries = [self.country_name for i in range(self.num_features)]
            return pd.DataFrame.from_records(self.feature_dict, index=countries)

    def get_labels(self, data_type):
        """
        Return labels as specified by argument
        :param data_type DataType either DICT or PANDAS_DF
        :rtype: python dictionary or pandas data frame
        """
        if not self.label_dict:
            for index, row in self.carbon_df.iterrows():
                if "MTCO2/day" in row['TOTAL_CO2_MED']:
                    continue
                date = utils.conform_date(row['DATE'])
                self.label_dict[date] = row['TOTAL_CO2_MED']

        if data_type == DataType.DICT:
            return self.label_dict
        if data_type == DataType.PANDAS_DF:
            return pd.DataFrame.from_records(self.label_dict, index=[0])

    def get_augmented_data(self, data_type):
        """
        Return features as specified by argument
        :param data_type DataType either DICT or PANDAS_DF
        :rtype: python dictionary or pandas data frame. For dictionary return type, first argument will be country name
        followed by dictionary of augmented data.
        """
        if not self.combined_data_dict:
            if not self.feature_dict:
                self.get_features(DataType.DICT)

            if not self.label_dict:
                self.get_labels(DataType.DICT)

            min_entries = len(self.label_dict) if (len(self.feature_dict) > len(self.label_dict)) else \
                len(self.feature_dict)
            iterable_dict = self.label_dict if (min_entries == len(self.label_dict)) else self.feature_dict

            for key in iterable_dict:
                features = self.feature_dict[key]
                label = self.label_dict[key]
                features.append(label)
                self.combined_data_dict[key] = features

        if data_type == DataType.DICT:
            return [self.country_name, self.combined_data_dict]

        if data_type == DataType.PANDAS_DF:
            countries = [self.country_name for i in range(self.num_features + 1)]
            return pd.DataFrame.from_records(self.combined_data_dict, index=countries)

    def save_data_frame_to_csv(self, location):
        """
        Method saves data frame to csv to location provided by the argument.
        :param location: Location of output csv on local file system.
        """
        pass

    @staticmethod
    def policy_values_as_list(data_row):
        """
        Return 17 policy parameters along with flags in relevant fields as list
        :rtype: List of policy parameters
        """
        c1 = data_row['C1_School closing']
        c1_flag = data_row['C1_Flag']
        c2 = data_row['C2_Workplace closing']
        c2_flag = data_row['C2_Flag']
        c3 = data_row['C3_Cancel public events']
        c3_flag = data_row['C3_Flag']
        c4 = data_row['C4_Restrictions on gatherings']
        c4_flag = data_row['C4_Flag']
        c5 = data_row['C5_Close public transport']
        c5_flag = data_row['C5_Flag']
        c6 = data_row['C6_Stay at home requirements']
        c6_flag = data_row['C6_Flag']
        c7 = data_row['C7_Restrictions on internal movement']
        c7_flag = data_row['C7_Flag']
        c8 = data_row['C8_International travel controls']
        e1 = data_row['E1_Income support']
        e1_flag = data_row['E1_Flag']
        e2 = data_row['E2_Debt/contract relief']
        e3 = data_row['E3_Fiscal measures']
        e4 = data_row['E4_International support']
        h1 = data_row['H1_Public information campaigns']
        h1_flag = data_row['H1_Flag']
        h2 = data_row['H2_Testing policy']
        h3 = data_row['H3_Contact tracing']
        h4 = data_row['H4_Emergency investment in healthcare']
        h5 = data_row['H5_Investment in vaccines']
        return [c1, c1_flag, c2, c2_flag, c3, c3_flag, c4, c4_flag, c5,
                c5_flag, c6, c6_flag, c7, c7_flag, c8, e1, e1_flag, e2, e3, e4, h1,
                h1_flag, h2, h3, h4, h5]
