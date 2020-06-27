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

    @staticmethod
    def policy_values_as_list(data_row):
        """
        Return 17 policy parameters along with flags in relevant fields as list
        :rtype: List of policy parameters
        """
        c1 = data_row[PolicyData.get_c1()]
        c1_flag = data_row[PolicyData.get_c1_flag()]
        c2 = data_row[PolicyData.get_c2()]
        c2_flag = data_row[PolicyData.get_c2_flag()]
        c3 = data_row[PolicyData.get_c3()]
        c3_flag = data_row[PolicyData.get_c3_flag()]
        c4 = data_row[PolicyData.get_c4()]
        c4_flag = data_row[PolicyData.get_c4_flag()]
        c5 = data_row[PolicyData.get_c5()]
        c5_flag = data_row[PolicyData.get_c5_flag()]
        c6 = data_row[PolicyData.get_c6()]
        c6_flag = data_row[PolicyData.get_c6_flag()]
        c7 = data_row[PolicyData.get_c7()]
        c7_flag = data_row[PolicyData.get_c7_flag()]
        c8 = data_row[PolicyData.get_c8()]
        e1 = data_row[PolicyData.get_e1()]
        e1_flag = data_row[PolicyData.get_e1_flag()]
        e2 = data_row[PolicyData.get_e2()]
        e3 = data_row[PolicyData.get_e3()]
        e4 = data_row[PolicyData.get_e4()]
        h1 = data_row[PolicyData.get_h1()]
        h1_flag = data_row[PolicyData.get_h1_flag()]
        h2 = data_row[PolicyData.get_h2()]
        h3 = data_row[PolicyData.get_h3()]
        h4 = data_row[PolicyData.get_h4()]
        h5 = data_row[PolicyData.get_h5()]
        return [c1, c1_flag, c2, c2_flag, c3, c3_flag, c4, c4_flag, c5,
                c5_flag, c6, c6_flag, c7, c7_flag, c8, e1, e1_flag, e2, e3, e4, h1,
                h1_flag, h2, h3, h4, h5]


class PolicyData:

    @staticmethod
    def get_c1():
        return 'C1_School closing'

    @staticmethod
    def get_c1_flag():
        return 'C1_Flag'

    @staticmethod
    def get_c2():
        return 'C2_Workplace closing'

    @staticmethod
    def get_c2_flag():
        return 'C2_Flag'

    @staticmethod
    def get_c3():
        return 'C3_Cancel public events'

    @staticmethod
    def get_c3_flag():
        return 'C3_Flag'

    @staticmethod
    def get_c4():
        return 'C4_Restrictions on gatherings'

    @staticmethod
    def get_c4_flag():
        return 'C4_Flag'

    @staticmethod
    def get_c5():
        return 'C5_Close public transport'

    @staticmethod
    def get_c5_flag():
        return 'C5_Flag'

    @staticmethod
    def get_c6():
        return 'C6_Stay at home requirements'

    @staticmethod
    def get_c6_flag():
        return 'C6_Flag'

    @staticmethod
    def get_c7():
        return 'C7_Restrictions on internal movement'

    @staticmethod
    def get_c7_flag():
        return 'C7_Flag'

    @staticmethod
    def get_c8():
        return 'C8_International travel controls'

    @staticmethod
    def get_e1():
        return 'E1_Income support'

    @staticmethod
    def get_e1_flag():
        return 'E1_Flag'

    @staticmethod
    def get_e2():
        return 'E2_Debt/contract relief'

    @staticmethod
    def get_e3():
        return 'E3_Fiscal measures'

    @staticmethod
    def get_e4():
        return 'E4_International support'

    @staticmethod
    def get_h1():
        return 'H1_Public information campaigns'

    @staticmethod
    def get_h1_flag():
        return 'H1_Flag'

    @staticmethod
    def get_h2():
        return 'H2_Testing policy'

    @staticmethod
    def get_h3():
        return 'H3_Contact tracing'

    @staticmethod
    def get_h4():
        return 'H4_Emergency investment in healthcare'

    @staticmethod
    def get_h5():
        return 'H5_Investment in vaccines'


