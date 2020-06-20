"""
Created by: Tapan Sharma
Date: 20/06/20
"""
import logging
import os

from unittest import TestCase
from predictCO2.preprocessing.generate_data import CountryPolicyCarbonData, DataType

dir_path = os.path.dirname(os.path.realpath(__file__))

carbon_csv_loc = dir_path + "/test_csv_files/label.csv"
policy_csv_loc = dir_path + "/test_csv_files/feature.csv"

logging.getLogger().setLevel(logging.INFO)


class TestCountryPolicyCarbonData(TestCase):

    def test_get_features(self):
        """
            1. Check data type was created successfully
            2. Check expected dimensionality of data type
        """
        logging.info("Testing features!")
        data = CountryPolicyCarbonData(policy_csv_loc, carbon_csv_loc)
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
        data = CountryPolicyCarbonData(policy_csv_loc, carbon_csv_loc)
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
        data = CountryPolicyCarbonData(policy_csv_loc, carbon_csv_loc)
        augmented_data = data.get_augmented_data(DataType.DICT)
        augmented_data_df = data.get_augmented_data(DataType.PANDAS_DF)

        self.assertTrue(augmented_data)
        self.assertEqual(len(augmented_data), 2)
        self.assertEqual(len(augmented_data[1]), 163)

        self.assertTrue(not augmented_data_df.empty)
        self.assertEqual(len(augmented_data_df.index), 27)
        self.assertEqual(len(augmented_data_df.columns), 163)
