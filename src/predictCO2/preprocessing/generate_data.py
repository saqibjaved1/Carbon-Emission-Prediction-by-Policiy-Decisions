"""
Created by: Tapan Sharma
Date: 20/6/2020
"""

import abc
import Globals
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

    @abc.abstractmethod
    def save_data_frame_to_npz(self, location: str):
        """
        Save data frame as compressed numpy arrays
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

    def save_data_frame_to_npz(self, location: str):
        """
        Save data frame as compressed numpy arrays to location provided by the argument.
        :param location: Location of output npz on local file system
        """
        pass


class PolicyData:
    C1 = 'C1_School closing'
    C1_FLAG = 'C1_Flag'
    C2 = 'C2_Workplace closing'
    C2_FLAG = 'C2_Flag'
    C3 = 'C3_Cancel public events'
    C3_FLAG = 'C3_Flag'
    C4 = 'C4_Restrictions on gatherings'
    C4_FLAG = 'C4_Flag'
    C5 = 'C5_Close public transport'
    C5_FLAG = 'C5_Flag'
    C6 = 'C6_Stay at home requirements'
    C6_FLAG = 'C6_Flag'
    C7 = 'C7_Restrictions on internal movement'
    C7_FLAG = 'C7_Flag'
    C8 = 'C8_International travel controls'
    E1 = 'E1_Income support'
    E1_FLAG = 'E1_Flag'
    E2 = 'E2_Debt/contract relief'
    E3 = 'E3_Fiscal measures'
    E4 = 'E4_International support'
    H1 = 'H1_Public information campaigns'
    H1_FLAG = 'H1_Flag'
    H2 = 'H2_Testing policy'
    H3 = 'H3_Contact tracing'
    H4 = 'H4_Emergency investment in healthcare'
    H5 = 'H5_Investment in vaccines'

    # POLICY_DATA_FRAME_FULL = None

    def __init__(self, policy_csv, country):
        """
        Policy/Feature Data set is available as a merged excel file with each country having it's individual sheet named
        after that country.
        :param self.__country_name: Name of country
        :param self.__country_policy_dict: Dictionary of policy
        :param self.__country_policy_df: Data frame of policy
        """
        self.__country_name = country
        self.__country_policy_dict = {}
        self.__country_policy_df = pd.read_excel(policy_csv, sheet_name=country)
        self.__set_properties()

    def __set_properties(self):
        """
        Private method which filters the policy data and sets only policy keys.
        """
        if not self.__country_policy_dict:
            for index, row in self.__country_policy_df.iterrows():
                date = row['Date']
                c1 = row[PolicyData.C1]
                c1_flag = row[PolicyData.C1_FLAG]
                c2 = row[PolicyData.C2]
                c2_flag = row[PolicyData.C2_FLAG]
                c3 = row[PolicyData.C3]
                c3_flag = row[PolicyData.C3_FLAG]
                c4 = row[PolicyData.C4]
                c4_flag = row[PolicyData.C4_FLAG]
                c5 = row[PolicyData.C5]
                c5_flag = row[PolicyData.C5_FLAG]
                c6 = row[PolicyData.C6]
                c6_flag = row[PolicyData.C6_FLAG]
                c7 = row[PolicyData.C7]
                c7_flag = row[PolicyData.C7_FLAG]
                c8 = row[PolicyData.C8]
                e1 = row[PolicyData.E1]
                e1_flag = row[PolicyData.E1_FLAG]
                e2 = row[PolicyData.E2]
                e3 = row[PolicyData.E3]
                e4 = row[PolicyData.E4]
                h1 = row[PolicyData.H1]
                h1_flag = row[PolicyData.H1_FLAG]
                h2 = row[PolicyData.H2]
                h3 = row[PolicyData.H3]
                h4 = row[PolicyData.H4]
                h5 = row[PolicyData.H5]
                self.__country_policy_dict[str(date)] = [c1, c1_flag, c2, c2_flag, c3, c3_flag, c4, c4_flag, c5,
                                                         c5_flag, c6, c6_flag, c7, c7_flag, c8, e1, e1_flag, e2, e3, e4,
                                                         h1,
                                                         h1_flag, h2, h3, h4, h5]

    def get_country_policy_data(self, data_type):
        """
        Method to fetch policy/feature data for country
        :param data_type: Specifies the format of return data type
        :return: Policy data frame or dictionary for country
        """
        if data_type == DataType.DICT:
            return self.__country_policy_dict

        if data_type == DataType.PANDAS_DF:
            return self.__country_policy_df

    def get_specific_policy_data(self, policy_key, data_type):
        """
        Method to return policy for specific policy parameter (e.g. C1, C2, etc....)
        :param policy_key: Name of policy parameter
        :param data_type: Specifies the format of return data type
        :return: Policy specific data frame or dictionary for country
        """
        policy_values_series = self.__country_policy_df[policy_key]
        date_series = self.__country_policy_df['Date'].astype(str)
        dates = date_series.to_list()
        frame = policy_values_series.to_frame()
        if data_type == DataType.PANDAS_DF:
            frame.index = dates
            return frame
        if data_type == DataType.DICT:
            policy_dict = frame.to_dict()[policy_key]
            return policy_dict

    @property
    def get_country_name(self):
        """
        Method to  return country name
        :return: Country name string
        """
        return self.__country_name



    @staticmethod
    def get_h5():
        return 'H5_Investment in vaccines'


