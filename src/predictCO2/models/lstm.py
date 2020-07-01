"""
Author: Saqib Javed
Date: 01/7/2020
"""

from src.predictCO2.models import nn_template
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense


class Lstm(nn_template.NN_Template):
    def __init__(self, config):
        super(Lstm, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(None,None)))
        self.model.add(Dense(9))
        self.model.compile(
              loss=self.config["model"]["loss"],
              optimizer=self.config["model"]["loss"],
              metrics=self.config["model"]['metrics'])
