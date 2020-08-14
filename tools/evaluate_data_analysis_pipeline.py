"""
Created by: Tapan Sharma
Date: 13/08/20
"""
import argparse
import json
import pandas as pd
import sys
import time
import numpy as np
from dataset.countries_label import countries_labels
from predictCO2.models.forecast import naive_forecast, soft_accuracy
from predictCO2.models.deep_learning_model import DeepLearningModel
from predictCO2.models.mixed_nn_model import nn_model
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
    print("\nLOADING TRAINING CONFIGURATION..................")
    with open('cfg/eval_script_config.json') as f:
        config_file = json.load(f)
    training_config_cnn = json.load(open(config_file["training_config_cnn"]))
    training_config_dnn = json.load(open(config_file["training_config_dnn"]))
    training_config_lstm = json.load(open(config_file["training_config_lstm"]))
    print("\nTRAINING CONFIGURATION LOADED SUCCESSFULLY..................")
    norm_data = training_config_cnn['training']['normalize']

    # 2. Data Processing (split into train & test as specified by the percentage in the configuration file) and perform
    # Min-Max feature scaling. Full feature space is considered i.e. Social, Economic, Health policy indicators.
    print("PRE-PROCESSING.........................")
    countryPolicyCarbonData = CountryPolicyCarbonData('training_data.yaml', country, include_flags=False,
                                                      policy_category=PolicyCategory.SOCIAL_INDICATORS,
                                                      normalize=0)
    print("PRE-PROCESSING COMPLETE.........................")

    # Just get test data for evaluating a pre-trained model. The test data obtained is previously unseen to the training
    # data
    features = pd.DataFrame()
    labels = pd.DataFrame()
    train_x, train_y, test_x, test_y = countryPolicyCarbonData.split_train_test(fill_nan=True)
    features = features.append(test_x)
    labels = labels.append(test_y)

    # 3. Data Modeling (load pre-trained model for evaluation). Social Policy indicators is a
    # 8-D data matrix as a time series.
    print("LOADING MODEL.........................")
    cnn = DeepLearningModel(training_config_cnn, num_features=8, num_outputs=1)
    cnn.load()
    lstm = DeepLearningModel(training_config_lstm, num_features=8, num_outputs=1)
    lstm.load()
    dnn = nn_model(training_config_dnn, num_features=8, num_outputs=1)
    dnn.load()
    print("MODEL LOADED SUCCESSFULLY.........................")

    # 4. Evaluate the trained model. Calculate prediction latency. Get prediction MSE, MAE and Soft Accuracy.
    test_start = time.time_ns()
    co2 = train_y.values.tolist() + test_y.values.tolist()
    forecast = naive_forecast(co2, 1)
    test_end = time.time_ns()
    mse = np.mean(np.square(np.array(co2[1:]) - np.array(forecast)))
    mae = np.mean(np.abs(np.array(co2[1:]) - np.array(forecast)))
    soft_acc = soft_accuracy(np.array(co2[1:]), np.array(forecast))
    print("------------------------------------- NAIVE FORECAST ------------------------------------")
    print("TESTING TIME: {}".format(test_end - test_start))
    print("\n\nTesting MSE: {}\nTesting Soft Accuracy: {}\nTesting MAE: {}".format(mse, soft_acc, mae))

    test_f, test_l = utils.data_sequence_generator(features, labels, training_config_cnn['time_steps'])
    test_start = time.time()
    model_eval = dnn.model.evaluate(test_f, test_l)
    test_end = time.time()
    print("------------------------------------- DNN MODEL ------------------------------------")
    print(dnn.model.summary())
    print("\n\nPREDICTION LATENCY: {}s".format(test_end - test_start))
    print("\n\nMSE: {}".format(model_eval[0]))


    test_start = time.time()
    model_eval = lstm.model.evaluate(test_f, test_l)
    test_end = time.time()
    print("------------------------------------- LSTM MODEL ------------------------------------")
    print(lstm.model.summary())
    print("TESTING TIME: {}".format(test_end - test_start))
    print("\n\nMSE: {}\nSoft Accuracy: {}\nMAE: {}".format(model_eval[0], model_eval[1],
                                                                                   model_eval[2]))

    test_start = time.time()
    model_eval = cnn.model.evaluate(test_f, test_l)
    test_end = time.time()
    print("------------------------------------- CNN MODEL ------------------------------------")
    print(cnn.model.summary())
    print("TESTING TIME: {}".format(test_end - test_start))
    print("\n\nMSE: {}\nSoft Accuracy: {}\nMAE: {}".format(model_eval[0], model_eval[1],
                                                                                   model_eval[2]))