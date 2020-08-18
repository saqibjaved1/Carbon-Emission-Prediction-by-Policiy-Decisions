"""
Created by: Tapan Sharma
Date: 13/08/20
"""

import json
from tensorflow import keras
from predictCO2.models.deep_learning_model import DeepLearningModel
from predictCO2.models.mixed_nn_model import nn_model
from predictCO2.models.lstm import Lstm

# 1. LSTM_TAPAN
with open('cfg/lstm_config.json') as f:
    lstm_config = json.load(f)

lstm = DeepLearningModel(lstm_config, num_features=17, num_outputs=1)
lstm.load()
opti = lstm.model.optimizer
print("LSTM_TAPAN\n\n")
print("Learning Rate: {}\n".format(keras.backend.eval(opti.lr)))
print("Beta 1: {}\n".format(keras.backend.eval(opti.beta_1)))
print("Beta 2: {}\n".format(keras.backend.eval(opti.beta_2)))
print("Epsilon: {}\n".format(keras.backend.eval(opti.epsilon)))
print("Amsgrad: {}\n\n\n".format(keras.backend.eval(opti.amsgrad)))

# # 2. CNN_LSTM_TOAHA
# with open('cfg/cnn_lstm_config.json') as f:
#     cnn_lstm_config = json.load(f)
#
# cnn_lstm = DeepLearningModel(cnn_lstm_config, num_features=17, num_outputs=1)
# cnn_lstm.load()
# opti = cnn_lstm.model.optimizer
# print("CNN_LSTM_TOAHA\n\n")
# print("Learning Rate: {}\n".format(keras.backend.eval(opti.lr)))
# print("Beta 1: {}\n".format(keras.backend.eval(opti.beta_1)))
# print("Beta 2: {}\n".format(keras.backend.eval(opti.beta_2)))
# print("Epsilon: {}\n".format(keras.backend.eval(opti.epsilon)))
# print("Amsgrad: {}\n\n\n".format(keras.backend.eval(opti.amsgrad)))

# 3. CNN_TAPAN
with open('cfg/cnn_config.json') as f:
    lstm_config = json.load(f)

lstm = DeepLearningModel(lstm_config, num_features=17, num_outputs=1)
lstm.load()
opti = lstm.model.optimizer
print("CNN_TAPAN\n\n")
print("Learning Rate: {}\n".format(keras.backend.eval(opti.lr)))
print("Beta 1: {}\n".format(keras.backend.eval(opti.beta_1)))
print("Beta 2: {}\n".format(keras.backend.eval(opti.beta_2)))
print("Epsilon: {}\n".format(keras.backend.eval(opti.epsilon)))
print("Amsgrad: {}\n\n\n".format(keras.backend.eval(opti.amsgrad)))

# 4. DNN_SREETAMA
with open('cfg/densenn_config.json') as f:
    dnn_config = json.load(f)

dnn = nn_model(dnn_config, num_features=8, num_outputs=1)
dnn.load()
opti = dnn.model.optimizer
print("DNN_Sreetama\n\n")
print("Learning Rate: {}\n".format(keras.backend.eval(opti.lr)))
print("Beta 1: {}\n".format(keras.backend.eval(opti.beta_1)))
print("Beta 2: {}\n".format(keras.backend.eval(opti.beta_2)))
print("Epsilon: {}\n".format(keras.backend.eval(opti.epsilon)))
print("Amsgrad: {}\n\n\n".format(keras.backend.eval(opti.amsgrad)))

# 4. LSTM_SAQIB
with open('cfg/config.json') as f:
    dnn_config = json.load(f)

lstm_1 = Lstm(dnn_config)
lstm_1.load()
opti = lstm_1.model.optimizer
print("LSTM_SAQIB\n\n")
print("Learning Rate: {}\n".format(keras.backend.eval(opti.lr)))
print("Beta 1: {}\n".format(keras.backend.eval(opti.beta_1)))
print("Beta 2: {}\n".format(keras.backend.eval(opti.beta_2)))
print("Epsilon: {}\n".format(keras.backend.eval(opti.epsilon)))
print("Amsgrad: {}\n\n\n".format(keras.backend.eval(opti.amsgrad)))