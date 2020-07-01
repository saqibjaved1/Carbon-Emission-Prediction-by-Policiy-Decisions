"""
Created by: Tapan Sharma
Date: 20/06/20
"""
import logging
import os
import Globals

from unittest import TestCase
from predictCO2.preprocessing.generate_data import CountryPolicyCarbonData, DataType, PolicyData, CarbonEmissionData

carbon_csv_loc = Globals.ROOT_DIR + "/dataset/labels/Modified_Emission_Data.xlsx"
policy_csv_loc = Globals.ROOT_DIR + "/dataset/features/Modified_Stringency_Data.xlsx"

cfg = "training_data.yaml"
country = "Germany"

logging.getLogger().setLevel(logging.INFO)


class TestCountryPolicyCarbonData(TestCase):

    def test_get_features(self):
        """
            1. Check data type was created successfully
            2. Check expected dimensionality of data type
        """
        logging.info("Testing features!")
        data = CountryPolicyCarbonData(cfg, country)
        features = data.get_features(DataType.DICT)
        features_df = data.get_features(DataType.PANDAS_DF)

        self.assertTrue(features)
        self.assertEqual(len(features), 171)

        self.assertTrue(not features_df.empty)
        self.assertEqual(len(features_df.index), 26)
        self.assertEqual(len(features_df.columns), 171)

    def test_get_labels(self):
        """
            1. Check data type was created successfully
            2. Check expected dimensionality of data type
        """
        logging.info("Testing labels!")
        data = CountryPolicyCarbonData(cfg, country)
        labels = data.get_labels(DataType.DICT)
        labels_df = data.get_labels(DataType.PANDAS_DF)

        self.assertTrue(labels)
        self.assertEqual(len(labels), 163)

        self.assertTrue(not labels_df.empty)
        self.assertEqual(len(labels_df.index), 1)
        self.assertEqual(len(labels_df.columns), 163)

    def test_get_augmented_data(self):
        """
            1. Check data type was created successfully
            2. Check expected dimensionality of data type
        """
        logging.info("Testing augmented data!")
        data = CountryPolicyCarbonData(cfg, country)
        augmented_data = data.get_augmented_data(DataType.DICT)
        augmented_data_df = data.get_augmented_data(DataType.PANDAS_DF)

        self.assertTrue(augmented_data)
        self.assertEqual(len(augmented_data), 2)
        self.assertEqual(len(augmented_data[1]), 163)

        self.assertTrue(not augmented_data_df.empty)
        self.assertEqual(len(augmented_data_df.index), 27)
        self.assertEqual(len(augmented_data_df.columns), 163)


class TestPolicyData(TestCase):
    def test_get_country_policy_data(self):
        data = PolicyData(policy_csv_loc, country)
        data_dict = data.get_country_policy_data(DataType.DICT)
        data_frame = data.get_country_policy_data(DataType.PANDAS_DF)

        self.assertTrue(data_dict)
        self.assertEqual(len(data_dict), 171)

        self.assertTrue(not data_frame.empty)
        self.assertEqual(len(data_frame.columns), 43)
        self.assertEqual(len(data_frame.index), 171)

    def test_get_specific_policy_data(self):
        data = PolicyData(policy_csv_loc, country)
        data_dict_c1 = data.get_specific_policy_data(PolicyData.C1, DataType.DICT)
        data_df_c1 = data.get_specific_policy_data(PolicyData.C1, DataType.PANDAS_DF)

        self.assertTrue(data_dict_c1)
        self.assertEqual(len(data_dict_c1), 171)

        self.assertTrue(not data_df_c1.empty)
        self.assertEqual(len(data_df_c1.columns), 1)
        self.assertEqual(len(data_df_c1.index), 171)

    def test_get_country_name(self):
        data = PolicyData(policy_csv_loc, country)
        self.assertEqual(data.get_country_name, country)


class TestCarbonEmissionData(TestCase):
    def test_get_country_carbon_emission_data(self):
        data = CarbonEmissionData(carbon_csv_loc, country)
        data_dict = data.get_country_carbon_emission_data(DataType.DICT)
        data_df = data.get_country_carbon_emission_data(DataType.PANDAS_DF)

        self.assertTrue(data_dict)
        self.assertEqual(len(data_dict), 163)

        self.assertTrue(not data_df.empty)
        self.assertEqual(len(data_df.index), 164)
        self.assertEqual(len(data_df.columns), 27)

    def test_get_country_name(self):
        data = CarbonEmissionData(carbon_csv_loc, country)
        self.assertEqual(country, data.get_country_name)
