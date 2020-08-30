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

with open('cfg/strindex_model_config.json') as f:
    training_config = json.load(f)

country = training_config['country']

policy_category = {
    "all": PolicyCategory.ALL,
    "social": PolicyCategory.SOCIAL_INDICATORS,
    "economic": PolicyCategory.ECONOMIC_INDICATORS,
    "health": PolicyCategory.HEALTH_INDICATORS,
    "total_stringency": PolicyCategory.STRINGENCY_INDEX
}
policy = policy_category[training_config['policy']]

# Collect data
countryPolicyCarbonData = CountryPolicyCarbonData('training_data.yaml', country, include_flags=False,
                                                  policy_category=policy,
                                                  normalize=0)
train_x, train_y, test_x, test_y = countryPolicyCarbonData.split_train_test(fill_nan=False, validation_percentage=0.3)

n_steps = training_config["n_steps"]
train_features, train_labels = generate_time_series_df(train_x, train_y, n_steps)
test_features, test_labels = generate_time_series_df(test_x, test_y, n_steps)

print(train_features.shape)
print(train_labels.shape)
print(test_features.shape)
print(test_labels.shape)

#Scale data
xscaler = MinMaxScaler()
yscaler = MinMaxScaler()
X_train_scaled = xscaler.fit_transform(train_features)
X_test_scaled = xscaler.transform(test_features)
y_train_scaled = yscaler.fit_transform(train_labels)
y_test_scaled = yscaler.transform(test_labels)

#Linear Regression model fit
train_start = time.time_ns()
lr = LinearRegression()
lr.fit(train_features, train_y.iloc[n_steps:, :])
train_time = time.time_ns() - train_start

# Test Set
pred_start = time.time_ns()
y_pred_train = yscaler.inverse_transform(lr.predict(X_train_scaled))
y_pred_test = yscaler.inverse_transform(lr.predict(X_test_scaled))
pred_time = time.time_ns() - pred_start

print("MSE training fit: %.03f" %mean_squared_error(train_labels,y_pred_train))
print("R2 training fit: %.03f " %r2_score(train_labels,y_pred_train))
# print("Soft accuracy for train set: %.03f" %soft_accuracy(train_labels,y_pred_train, prediction_tolerance))
print("MSE prediction: %.03f" %mean_squared_error(test_labels,y_pred_test))
print("R2 prediction: %.03f " %r2_score(test_labels,y_pred_test))
# print("Soft accuracy for test set: %.03f" %soft_accuracy(test_labels,y_pred_test, prediction_tolerance))
print("Train time: {} ns, Prediction time: {} ns".format(train_time, pred_time))

#Visualize test set predicted and original labels
fig2, ax2 = plt.subplots()
ax2.plot(range(len(test_labels)), test_labels, label='True Labels')
ax2.plot(range(len(y_pred_test)), y_pred_test, label='Predicted Labels')
plt.xlabel('Time')
plt.ylabel('CO2 reduction')
plt.legend()
plt.title('Test Data')
plt.show()

# Future Prediction
pred_steps = training_config['prediction steps']
str_idx = input('Enter Stringency Index: ')
output_arr = []
input_data = np.zeros((1,train_features.shape[1]))
input_data[0, 0] = str_idx
for i in range(pred_steps):
    if i >= n_steps:
        input_data[0, 1:] = output_arr[i-n_steps:]
    out_data = lr.predict(input_data)
    output_arr.append(out_data)

#Visualize the predicted values for future
data_type = DataType.PANDAS_DF
co2_data_avlbl = countryPolicyCarbonData.get_labels(data_type).to_numpy().reshape(-1,1)
co2_data_pred = np.array(output_arr).reshape(-1,1)
co2_total_duration = np.concatenate((co2_data_avlbl, co2_data_pred))
fig2, ax2 = plt.subplots()
ax2.plot(range(len(co2_total_duration)), co2_total_duration, 'o')
plt.xlabel('Time')
plt.ylabel('CO2 reduction')
plt.legend()
plt.title('Test Data')
plt.show()