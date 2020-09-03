"""
Author: Sreetama Sarkar
Date: 8/30/2020
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from predictCO2.preprocessing.co2_percent_dict import co2_percentage
from predictCO2.preprocessing.generate_data import DataType, CountryPolicyCarbonData, PolicyCategory
from predictCO2.preprocessing.utils import generate_time_series_df
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class stringency_model():
    def __init__(self, country = 'India', stringency = 5, pred_steps = 30):
        self.country = country
        self.input_stringency = stringency
        with open('cfg/strindex_model_config.json') as f:
            self.training_config = json.load(f)
        self.n_steps = self.training_config["n_steps"]
        self.policy = self.training_config['policy']
        self.pred_steps = pred_steps

    def get_train_data(self):
        """
        Get country specific train and test data for training and evaluating model
        """
        policy_category = {
            "all": PolicyCategory.ALL,
            "social": PolicyCategory.SOCIAL_INDICATORS,
            "economic": PolicyCategory.ECONOMIC_INDICATORS,
            "health": PolicyCategory.HEALTH_INDICATORS,
            "total_stringency": PolicyCategory.STRINGENCY_INDEX
        }
        policy = policy_category[self.policy]

        # Collect data
        countryPolicyCarbonData = CountryPolicyCarbonData('training_data.yaml', self.country, include_flags=False,
                                                          policy_category=policy,
                                                          normalize=0)
        self.co2_data_avlbl = countryPolicyCarbonData.get_labels(DataType.PANDAS_DF)
        train_x, train_y, test_x, test_y = countryPolicyCarbonData.split_train_test(fill_nan=False,
                                                                                    validation_percentage=self.training_config['val_pc'])

        self.train_features, self.train_labels = generate_time_series_df(train_x, train_y, self.n_steps)
        self.test_features, self.test_labels = generate_time_series_df(test_x, test_y, self.n_steps)

        # Scale data
        # xscaler = MinMaxScaler()
        # yscaler = MinMaxScaler()
        # self.X_train_scaled = xscaler.fit_transform(train_features)
        # self.X_test_scaled = xscaler.transform(test_features)
        # self.y_train_scaled = yscaler.fit_transform(train_labels)
        # self.y_test_scaled = yscaler.transform(test_labels)


    def build_model(self):
        """
        Train regression model
        :return:
        """
        self.get_train_data()
        self.lr = LinearRegression()
        self.lr.fit(self.train_features, self.train_labels)


    def generate_future_prediction(self):
        """
        Generate predictions for future pred_steps
        """
        self.build_model()
        output_arr = []
        input_data = np.zeros((1, self.train_features.shape[1]))
        input_data[0, 0] = self.input_stringency
        input_data[0, 1:] = self.co2_data_avlbl.iloc[0, -self.n_steps:]
        for i in range(self.pred_steps):
            if i >= self.n_steps:
                input_data[0, 1:] = output_arr[i - self.n_steps:]
            out_data = self.lr.predict(input_data)
            output_arr.append(out_data)
        return output_arr


    def soft_accuracy(self, y_true, y_pred, tolerance):
        return np.mean(np.abs(y_true - y_pred) <= tolerance)


    def evaluate_model_performance(self):
        """
        Evaluate performance metrics for the model
        (For evaluating performance, increase validation_percentage (val_pc, configuration file) to 0.3)
        """
        self.build_model()
        prediction_tolerance = self.training_config['prediction_tolerance']
        # Test Set
        self.y_pred_train = self.lr.predict(self.train_features)
        self.y_pred_test = self.lr.predict(self.test_features)
        # y_pred_train = yscaler.inverse_transform(lr.predict(X_train_scaled))
        # y_pred_test = yscaler.inverse_transform(lr.predict(X_test_scaled))

        print("MSE training fit: %.05f" % mean_squared_error(self.train_labels, self.y_pred_train))
        print("R2 training fit: %.05f " % r2_score(self.train_labels, self.y_pred_train))
        # print("Soft accuracy for train set: %.03f" %self.soft_accuracy(self.train_labels, self.y_pred_train, prediction_tolerance))
        print("MSE prediction: %.05f" % mean_squared_error(self.test_labels, self.y_pred_test))
        print("R2 prediction: %.05f " % r2_score(self.test_labels, self.y_pred_test))
        # print("Soft accuracy for test set: %.03f" %self.soft_accuracy(self.test_labels,self.y_pred_test, prediction_tolerance))
        # print("Train time: {} ns, Prediction time: {} ns".format(train_time, pred_time))

    def visualize_predicted_vs_original(self):
        """
        Visualize test set predicted and original labels
        (For visualizing test predictions, increase validation_percentage (val_pc, configuration file) to 0.3)
        :return:
        """
        self.evaluate_model_performance()
        fig2, ax2 = plt.subplots()
        ax2.plot(range(len(self.test_labels)), self.test_labels, label='True Labels')
        ax2.plot(range(len(self.y_pred_test)), self.y_pred_test, label='Predicted Labels')
        plt.xlabel('Time')
        plt.ylabel('CO2 reduction')
        plt.legend()
        plt.title('Test Data')
        plt.show()


    def visualize_future_predictions(self):
        """
        Demo Visualizations for UI
        """
        countryPolicyCarbonData = CountryPolicyCarbonData('training_data.yaml', self.country, include_flags=False,
                                                          policy_category=PolicyCategory.STRINGENCY_INDEX,
                                                          normalize=0)

        data_type = DataType.PANDAS_DF
        co2_data_avlbl = countryPolicyCarbonData.get_labels(data_type)
        data_avlbl_dates = co2_data_avlbl.columns
        avlbl_dates = pd.to_datetime(data_avlbl_dates)
        next_dates = pd.date_range(data_avlbl_dates[-1], periods=self.pred_steps + 1)
        co2_data_pred = self.generate_future_prediction()
        co2_total_duration = np.concatenate((co2_data_avlbl.to_numpy().reshape(-1, 1), np.array(co2_data_pred).reshape(-1, 1)))
        dates = avlbl_dates.union(next_dates)
        fig2, ax2 = plt.subplots()
        ax2.plot(dates, co2_total_duration, 'o')

        plt.xlabel('Time')
        plt.ylabel('CO2 reduction')
        plt.legend()
        plt.title('Test Data')
        plt.show()


# Test
# str_model = stringency_model(country = 'Italy', stringency= 98)
# str_model.visualize_predicted_vs_original()
# str_model.visualize_future_predictions()
