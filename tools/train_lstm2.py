"""
Created by: Tapan Sharma
Date: 14/07/20
"""
import json

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from predictCO2.models.lstm_2 import LSTM_2
from predictCO2.preprocessing import utils
from predictCO2.preprocessing.generate_data import CountryPolicyCarbonData

with open('cfg/lstm_config.json') as f:
    training_config = json.load(f)

countries = training_config['countries']
train_features = pd.DataFrame()
train_labels = pd.DataFrame()
test_features = pd.DataFrame()
test_labels = pd.DataFrame()

# Collect data
for country in countries:
    countryPolicyCarbonData = CountryPolicyCarbonData('training_data.yaml', country)
    train_x, train_y, test_x, test_y = countryPolicyCarbonData.split_train_test(fill_nan=True)
    train_features = train_features.append(train_x)
    test_features = test_features.append(test_x)
    train_labels = train_labels.append(train_y)
    test_labels = test_labels.append(test_y)