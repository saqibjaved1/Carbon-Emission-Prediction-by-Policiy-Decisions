"""
Created by: Tapan Sharma
Date: 20/6/2020
"""

import abc
import math
from abc import ABC
import logging
import Globals
import pandas as pd
import predictCO2.preprocessing.utils as utils

from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class DataType(Enum):
    """
    Enum specifying type in which training data is to be made available.
    """
    DICT = 1  # For type dictionary
    PANDAS_DF = 2  # For type pandas data frame


class PolicyCategory(Enum):
    ALL = 1
    SOCIAL_INDICATORS = 2
    ECONOMIC_INDICATORS = 3
    HEALTH_INDICATORS = 4
    STRINGENCY_INDEX = 5

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


class CountryPolicyCarbonData(TrainDataInterface, ABC):
    def __init__(self, training_cfg, country, policy_category=PolicyCategory.ALL, include_flags=True, normalize=0):
        """
        Class to keep Policy and Carbon data by country.
        Uses OXFORD OxCGRT (https://github.com/OxCGRT/covid-policy-tracker) as features/policy
        Uses Global Carbon Project (https://www.icos-cp.eu/gcp-covid19) as labels/carbon
        :param training_cfg: yaml configuration file where location of data set and other parameters as kept
        :param country: name of country for which data is required.
        """
        self.training_cfg = utils.load_cfg_file(training_cfg)
        self.country_name = country
        carbon_csv = Globals.ROOT_DIR + "/" + self.training_cfg['labels']
        self.carbon_emission_data = CarbonEmissionData(carbon_csv, country)

        policy_csv = Globals.ROOT_DIR + "/" + self.training_cfg['features']
        self.policy_data = PolicyData(policy_csv, self.country_name, policy_category=policy_category,
                                      include_flags=include_flags)

        self.combined_data_df = pd.DataFrame()
        self.feature_df = pd.DataFrame()
        self.label_df = pd.DataFrame()
        self.num_features = 0
        self.num_timestamps = 0
        self.test_feature_df = pd.DataFrame()
        self.test_label_df = pd.DataFrame()
        self.normalize = normalize

    def get_features(self, data_type, fill_na=True):
        """
        Return features as specified by argument
        :param data_type DataType either DICT or PANDAS_DF
        :rtype: python dictionary or pandas data frame
        """
        if self.feature_df.empty:
            self.feature_df, _ = self.__get_conformed_data()

        if fill_na:
            self.feature_df = self.feature_df.fillna(0)

        if data_type == DataType.DICT:
            return self.feature_df.to_dict()
        if data_type == DataType.PANDAS_DF:
            return self.feature_df

    def get_labels(self, data_type, fill_na=True):
        """
        Return labels as specified by argument
        :param data_type DataType either DICT or PANDAS_DF
        :rtype: python dictionary or pandas data frame
        """
        if self.label_df.empty:
            _, self.label_df = self.__get_conformed_data()
        if fill_na:
            self.label_df = self.label_df.fillna(0)
        if data_type == DataType.DICT:
            return self.label_df.to_dict()
        if data_type == DataType.PANDAS_DF:
            return self.label_df

    def __get_conformed_data(self):
        """
        Conforms sizes of features and labels so that they are equal.
        :rtype: feature and label data frames.
        """
        if self.feature_df.empty:
            self.feature_df = self.policy_data.get_country_policy_data(DataType.PANDAS_DF)
            self.num_features, self.num_timestamps = self.feature_df.shape

        if self.label_df.empty:
            self.label_df = self.carbon_emission_data.get_country_carbon_emission_data(DataType.PANDAS_DF)

        feat_shape = self.feature_df.shape
        lab_shape = self.label_df.shape

        # Equate sizes
        if feat_shape[1] > lab_shape[1]:
            self.feature_df = self.feature_df[self.feature_df.columns[0:lab_shape[1]]]
        else:
            self.label_df = self.label_df[self.label_df.columns[0:feat_shape[1]]]

        if self.normalize:
            self.feature_df = self.feature_df.sub(self.feature_df.min()).div(self.feature_df.max().
                                                                                     sub(self.feature_df.min()))
            # self.feature_df = self.feature_df.sub(self.feature_df.mean(1), axis=0).div(self.feature_df.std(1), axis=0)
            self.label_df = self.label_df.sub(self.label_df.min()).div(self.label_df.max().
                                                                               sub(self.label_df.min()))
            # self.label_df = self.label_df.sub(self.label_df.mean(1), axis=0).div(self.label_df.std(1), axis=0)
        return self.feature_df, self.label_df

    def split_train_test(self, validation_percentage=None, fill_nan=False):
        """
        Splits the data set into training and testing. (Learning process can further have a possibility to split data
        into training and validation. Hence, from this function, test data can be used as a new unseen data for model
        to be evaluated upon. Otherwise, test set generated can be used as explicit validation set during training.)
        :param validation_percentage: percentage of validation split
        :param fill_nan: If true replaces all NaNs with 0.
        :return: training and testing data sets.
        """
        if not validation_percentage:
            validation_percentage = self.training_cfg['val_pc']

        logger.info("Splitting data set for {} with test percentage: {}".format(self.country_name,
                                                                                validation_percentage))

        features_raw = self.get_features(DataType.PANDAS_DF)
        features_raw = features_raw.T
        labels_raw = self.get_labels(DataType.PANDAS_DF)
        labels_raw = labels_raw.T

        if fill_nan:
            features_raw = features_raw.fillna(0)
            labels_raw = labels_raw.fillna(0)

        samples = features_raw.shape[0]
        val_samples = math.ceil(samples * validation_percentage)
        train_features = features_raw.head(samples - val_samples)
        test_features = features_raw.tail(val_samples)
        train_labels = labels_raw.head(samples - val_samples)
        test_labels = labels_raw.tail(val_samples)

        return train_features, train_labels, test_features, test_labels

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
    STRINGENCY_INDEX = 'StringencyIndexForDisplay'

    # POLICY_DATA_FRAME_FULL = None

    def __init__(self, policy_csv, country, policy_category=PolicyCategory.ALL, include_flags=True):
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
        self.__policy_category = policy_category
        self.__include_flags = include_flags
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
                stringency = row[PolicyData.STRINGENCY_INDEX]

                if self.__include_flags:
                    if self.__policy_category == PolicyCategory.ALL:
                        self.__country_policy_dict[str(date)] = [c1, c1_flag, c2, c2_flag, c3, c3_flag, c4, c4_flag, c5,
                                                                 c5_flag, c6, c6_flag, c7, c7_flag, c8, e1, e1_flag, e2,
                                                                 e3, e4, h1, h1_flag, h2, h3, h4, h5]
                    elif self.__policy_category == PolicyCategory.SOCIAL_INDICATORS:
                        self.__country_policy_dict[str(date)] = [c1, c1_flag, c2, c2_flag, c3, c3_flag, c4, c4_flag, c5,
                                                                 c5_flag, c6, c6_flag, c7, c7_flag, c8]
                    elif self.__policy_category == PolicyCategory.ECONOMIC_INDICATORS:
                        self.__country_policy_dict[str(date)] = [e1, e1_flag, e2, e3, e4]
                    elif self.__policy_category == PolicyCategory.HEALTH_INDICATORS:
                        self.__country_policy_dict[str(date)] = [h1, h1_flag, h2, h3, h4, h5]
                    elif self.__policy_category == PolicyCategory.STRINGENCY_INDEX:
                        self.__country_policy_dict[str(date)] = [stringency]
                else:
                    if self.__policy_category == PolicyCategory.ALL:
                        self.__country_policy_dict[str(date)] = [c1, c2, c3, c4, c5, c6, c7, c8, e1, e2, e3, e4, h1, h2,
                                                                 h3, h4, h5]
                    elif self.__policy_category == PolicyCategory.SOCIAL_INDICATORS:
                        self.__country_policy_dict[str(date)] = [c1, c2, c3, c4, c5, c6, c7, c8]
                    elif self.__policy_category == PolicyCategory.ECONOMIC_INDICATORS:
                        self.__country_policy_dict[str(date)] = [e1, e2, e3, e4]
                    elif self.__policy_category == PolicyCategory.HEALTH_INDICATORS:
                        self.__country_policy_dict[str(date)] = [h1, h2, h3, h4, h5]
                    elif self.__policy_category == PolicyCategory.STRINGENCY_INDEX:
                        self.__country_policy_dict[str(date)] = [stringency]
            self.__country_policy_df = pd.DataFrame.from_records(self.__country_policy_dict)

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


class CarbonEmissionData:
    def __init__(self, carbon_csv, country):
        """
        Carbon emission data initializer.
        :param carbon_csv: Path to carbon emission excel
        :param country: country name
        """
        self.__carbon_df = pd.read_excel(carbon_csv, sheet_name=country)
        self.__country_name = country
        self.__carbon_emission_dict = {}
        self.__set_properties()

    def __set_properties(self):
        """
        Filters the Carbon emission file to select relevant keys/rows
        """
        if not self.__carbon_emission_dict:
            for index, row in self.__carbon_df.iterrows():
                if "MTCO2/day" in str(row['TOTAL_CO2_MED']):
                    continue
                date = utils.conform_date(row['DATE'])
                self.__carbon_emission_dict[str(date)] = [row['TOTAL_CO2_MED']]
        self.__carbon_df = pd.DataFrame.from_records(self.__carbon_emission_dict)

    def get_country_carbon_emission_data(self, data_type):
        """
        Provides carbon emission data as specified
        :param data_type: Specifies the format of return data type
        :return: either dict or data frame as specified
        """
        if data_type == DataType.DICT:
            return self.__carbon_emission_dict
        if data_type == DataType.PANDAS_DF:
            return self.__carbon_df

    @property
    def get_country_name(self):
        """
        Returns name of country
        :return: country name
        """
        return self.__country_name
