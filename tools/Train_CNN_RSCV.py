"""
Created by : Subarnaduti Paul
Date: 27/08/20
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.predictCO2.preprocessing.generate_data import CountryPolicyCarbonData, PolicyCategory
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer
from tensorflow.keras import layers, models, optimizers, backend
from src.predictCO2.models.RandomizedSearchCV_CNN import CNN
from src.predictCO2.preprocessing import utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, make_scorer
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

#learning rate
lr = [1e-2, 1e-3, 1e-4]

# activation
activation = ['relu', 'tanh']

#layers
nn1 = [32, 64, 96]
nn2 = [32, 64, 96]
dn1 = [50, 75, 100]
dn2 = [50, 80, 100]

# dropout and regularisation
dp = [0, 0.1, 0.2, 0.3]
decay = [1e-4, 1e-6, 0]
n_steps = config['time_steps']

# dictionary summary- Hyperparameters
param_grid = {'lr':[1e-2,1e-3,1e-4],
              'decay':[1e-5,1e-6,1e-7],
              'dp':[0.1,0.2,0.3],
              'nn1': [32, 64, 96],
              'nn2': [32, 64, 96],
              'dn1': [50, 75, 100],
              'dn2': [50, 80, 100]

}



conv = CNN(config,
           num_features=n_features, num_steps=config['time_steps'])
grid = conv.tuning(param_grid)

grid_result = grid.fit(feature, labels)
print(grid.best_params_)


# Model evaluation
# model_eval = Conv.model.evaluate(feature_t, labels_t)
# print("\n\nTesting Loss: {}\nTesting Accuracy: {}".format(model_eval[0], model_eval[1]))
# Conv.save("CNN_model")


