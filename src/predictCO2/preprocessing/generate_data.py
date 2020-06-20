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

