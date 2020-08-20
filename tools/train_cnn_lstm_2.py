# -*- coding: utf-8 -*-
# @Time    : 8/20/20 3:14 AM
# @Author  : Saptarshi
# @Email   : saptarshi.mitra@tum.de
# @File    : train_cnn_lstm_2.py
# @Project: group07


import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from predictCO2.models.deep_learning_model import DeepLearningModel
from predictCO2.preprocessing import utils
from predictCO2.preprocessing.generate_data import CountryPolicyCarbonData, PolicyCategory
from sklearn.model_selection import TimeSeriesSplit

matplotlib.use('Qt5Agg')

with open('cfg/cnn_lstm_2_config.json') as f:
    training_config = json.load(f)

countries = training_config['countries']
train_features = pd.DataFrame()
train_labels = pd.DataFrame()
test_features = pd.DataFrame()
test_labels = pd.DataFrame()
norm_data = training_config['training']['normalize']

# Collect data
for country in countries:
    countryPolicyCarbonData = CountryPolicyCarbonData('training_data.yaml', country, include_flags=False,
                                                      policy_category=PolicyCategory.ALL,
                                                      normalize=norm_data)
    train_x, train_y, test_x, test_y = countryPolicyCarbonData.split_train_test(fill_nan=False)
    train_features = train_features.append(train_x)
    test_features = test_features.append(test_x)
    train_labels = train_labels.append(train_y)
    test_labels = test_labels.append(test_y)

print(train_features.shape)
print(train_labels.shape)
print(test_features.shape)
print(test_labels.shape)
# print(train_labels.eq(0).all())
# Train model with 5 fold cross validation
tss = TimeSeriesSplit()
_, n_features = train_features.shape
cnn = DeepLearningModel(training_config, num_features=n_features, num_outputs=1)
print(cnn.model.summary())
cnn.plot_and_save_model("content/model_arch/CNN_LSTM_SAPTARSHI.png")
print(cnn.model.summary())
losses = []
start = time.time()
for train_idx, test_idx in tss.split(train_features):
    X, X_val = train_features.iloc[train_idx], train_features.iloc[test_idx]
    Y, Y_val = train_labels.iloc[train_idx], train_labels.iloc[test_idx]
    features, labels = utils.data_sequence_generator(X, Y, training_config['time_steps'])
    val_f, val_l = utils.data_sequence_generator(X_val, Y_val, training_config['time_steps'])
    h = cnn.train_with_validation_provided(features, labels, val_f, val_l)
    losses.append(h.history['loss'])
end = time.time()
print("TRAINING TIME: {}".format(end - start))

# Plot training loss
loss_arr = np.zeros((100, 1))
for loss_per_fold in losses:
    for j, loss in enumerate(loss_per_fold):
        loss_arr[j] = loss_arr[j] + loss
loss_arr = loss_arr / 5
fig1, ax1 = plt.subplots()
ax1.plot(range(len(loss_arr)), loss_arr, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.show()

# # # Prediction
test_start = time.time()
test_f, test_l = utils.data_sequence_generator(test_features, test_labels, training_config['time_steps'])
model_eval = cnn.model.evaluate(test_f, test_l)
y = cnn.model.predict(test_f)
print(y)
test_end = time.time()
print("TESTING TIME: {}".format(test_end - test_start))
print("\n\nTesting Loss: {}\nTesting Accuracy: {}".format(model_eval[0], model_eval[1]))
cnn.save("CNN_LSTM_2_SAPTARSHI")
