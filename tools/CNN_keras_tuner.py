"""
Created by : Subarnaduti Paul
Date: 27/08/20
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.predictCO2.preprocessing.generate_data import CountryPolicyCarbonData, PolicyCategory
import pandas as pd
from sklearn.metrics import  make_scorer
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from src.predictCO2.models.CNN_tuner import CNN
from src.predictCO2.preprocessing import utils
from sklearn.model_selection import train_test_split

import json


with open('cfg/CNN.json') as jsonfile:
    config = json.load(jsonfile)


def plot_model_history(history, ax=None, metric='loss', ep_start=1, ep_stop=None, monitor='val_loss', mode='min',
                       plttitle=None):
    if ax is None:
        fig, ax = plt.subplots()
    if ep_stop is None:
        ep_stop = len(history.epoch)
    if plttitle is None:
        plttitle = metric[0].swapcase() + metric[1:] + ' During Training'
    ax.plot(np.arange(ep_start, ep_stop + 1, dtype='int'), history.history[metric][ep_start - 1:ep_stop])
    ax.plot(np.arange(ep_start, ep_stop + 1, dtype='int'), history.history['val_' + metric][ep_start - 1:ep_stop])
    ax.set(title=plttitle)
    ax.set(ylabel=metric[0].swapcase() + metric[1:])
    ax.set(xlabel='Epoch')
    ax.legend(['train', 'val'], loc='upper right')
    plt.show()


countries = config["countries"]
train_features = pd.DataFrame()
train_labels = pd.DataFrame()
test_features = pd.DataFrame()
test_labels = pd.DataFrame()

# Fetching data for features and labels
for country in countries:
    countryPolicyCarbonData = CountryPolicyCarbonData("training_data.yaml", country, include_flags=False)
    # policy_category=PolicyCategory.SOCIAL_INDICATORS)

    train_x, train_y, test_x, test_y = countryPolicyCarbonData.split_train_test(fill_nan=True)
    train_features = train_features.append(train_x)
    test_features = test_features.append(test_x)
    train_labels = train_labels.append(train_y)
    test_labels = test_labels.append(test_y)

print(train_features.shape)
print(train_labels.shape)
print(test_features.shape)
print(test_labels.shape)

# Spliting datasets into training,validation and test set and time_series data generation
feature, labels = utils.time_series_data_generator(train_features, train_labels, config['time_steps'])
feature_t, labels_t = utils.time_series_data_generator(test_features, test_labels, config['time_steps'])
print(labels.shape)
x_train, x_val, y_train, y_val = train_test_split(feature, labels, test_size=0.15)

print(x_val.shape)
n_features = x_train.shape[2]
print(n_features)

#Performing Keras Tuner search
conv = CNN(config,
           num_features=n_features, num_steps=config['time_steps'])
tuner = conv.tuning()
tuner.search(x_train,y_train,epochs=30,validation_data=(x_val, y_val))

tuner.results_summary()

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model.
loss, accuracy = best_model.evaluate(feature_t, labels_t)
print(accuracy)




# Plot the training loss
# fig1, ax1 = plt.subplots()
# ax1.plot(range(len(loss)), loss, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss')
# plt.show()

# Conv.save("CNN_model")


