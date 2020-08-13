"""
Author: Sreetama Sarkar
Date: 8/10/2020
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from predictCO2.preprocessing.co2_percent_dict import co2_percentage
from predictCO2.preprocessing.generate_data import CountryPolicyCarbonData, PolicyCategory
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def soft_accuracy(y_true, y_pred, tolerance):
    return np.mean(np.abs(y_true - y_pred) <= tolerance)

with open('cfg/regression_config.json') as f:
    training_config = json.load(f)

countries = training_config['countries']
train_features = pd.DataFrame()
train_labels = pd.DataFrame()
test_features = pd.DataFrame()
test_labels = pd.DataFrame()

policy_category = {
    "all": PolicyCategory.ALL,
    "social": PolicyCategory.SOCIAL_INDICATORS,
    "economic": PolicyCategory.ECONOMIC_INDICATORS,
    "health": PolicyCategory.HEALTH_INDICATORS,
    "total_stringency": PolicyCategory.STRINGENCY_INDEX
}
policy = policy_category[training_config['policy']]
prediction_tolerance = training_config['prediction_tolerance']
# Collect data
for country in countries:
    countryPolicyCarbonData = CountryPolicyCarbonData('training_data.yaml', country, include_flags=False,
                                                      policy_category=policy,
                                                      normalize=0)
    train_x, train_y, test_x, test_y = countryPolicyCarbonData.split_train_test(fill_nan=False)
    # train_x['percent_contrib'] = co2_percentage[country]
    # test_x['percent_contrib'] = co2_percentage[country]
    train_features = train_features.append(train_x)
    test_features = test_features.append(test_x)
    train_labels = train_labels.append(train_y)
    test_labels = test_labels.append(test_y)

print(train_features.shape)
print(train_labels.shape)
print(test_features.shape)
print(test_labels.shape)

#Scale data
xscaler = StandardScaler()
yscaler = StandardScaler()
X_train_scaled = xscaler.fit_transform(train_features)
X_test_scaled = xscaler.transform(test_features)
y_train_scaled = yscaler.fit_transform(train_labels)
y_test_scaled = yscaler.transform(test_labels)

#Linear Regression model fit
train_start = time.time_ns()
lr = LinearRegression()
lr.fit(X_train_scaled,y_train_scaled)
train_time = time.time_ns() - train_start

#Prediction
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