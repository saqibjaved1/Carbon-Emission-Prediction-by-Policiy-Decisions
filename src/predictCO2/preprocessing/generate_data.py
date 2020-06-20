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


