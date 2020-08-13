"""
Created by: Tapan Sharma
Date: 13/08/20
"""
import argparse
import json
import pandas as pd
import sys
import time

from dataset.countries_label import countries_labels
from predictCO2.models.deep_learning_model import DeepLearningModel
from predictCO2.preprocessing import utils
from predictCO2.preprocessing.generate_data import CountryPolicyCarbonData, PolicyCategory

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(
        description='Evaluate Data Analysis Pipeline')
    parser.add_argument('--country', type=str, default='Germany',
                        help='Name of Country to run data analysis pipeline on ')
    args = parser.parse_args()

    country = args.country
    country = country.title()

    if country not in countries_labels:
        print("FAULT: {} not in the list for which reliable data is present.".format(country))
        print("Please select one from the following list: ")
        print(*countries_labels.keys(), sep=",")

    # 1. Load Configuration
    with open('cfg/cnn_config.json') as f:
        training_config = json.load(f)
    norm_data = training_config['training']['normalize']

    # 2. Data Processing (split into train & test as specified by the percentage in the configuration file) and perform
    # Min-Max feature scaling. Full feature space is considered i.e. Social, Economic, Health policy indicators.
    countryPolicyCarbonData = CountryPolicyCarbonData('training_data.yaml', country, include_flags=False,
                                                      policy_category=PolicyCategory.ALL,
                                                      normalize=norm_data)

    # Just get test data for evaluating a pre-trained model. The test data obtained is previously unseen to the training
    # data
    features = pd.DataFrame()
    labels = pd.DataFrame()
    _, _, test_x, test_y = countryPolicyCarbonData.split_train_test(fill_nan=True)
    features = features.append(test_x)
    labels = labels.append(test_y)

    # 3. Data Modeling (load pre-trained model for evaluation). Full feature space having all policy indicators is a
    # 17-D data matrix as a time series.
    cnn = DeepLearningModel(training_config, num_features=17, num_outputs=1)
    cnn.load()

    # 4. Evaluate the trained model. Calculate prediction latency. Get prediction MSE, MAE and Soft Accuracy.
    test_start = time.time()
    test_f, test_l = utils.data_sequence_generator(features, labels, training_config['time_steps'])
    model_eval = cnn.model.evaluate(test_f, test_l)
    test_end = time.time()
    print("TESTING TIME: {}".format(test_end - test_start))
    print("\n\nTesting MSE: {}\nTesting Soft Accuracy: {}\nTesting MAE: {}".format(model_eval[0], model_eval[1],
                                                                                   model_eval[2]))