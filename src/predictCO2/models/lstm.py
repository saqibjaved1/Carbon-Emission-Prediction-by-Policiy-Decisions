"""
Author: Saqib Javed
Date: 01/7/2020
"""
from src.predictCO2.models import nn_template
from src.predictCO2.preprocessing import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard


class Lstm(nn_template.NN_Template):
    def __init__(self, config):
        super(Lstm, self).__init__(config)
        self.build_model()

    def build_model(self):
        """
        Method to create the model
        """
        self.model = Sequential()
        self.model.add(LSTM(20, input_shape=(self.config['time_steps'], 8), return_sequences=True))
        self.model.add(Dropout(0.4))
        self.model.add(LSTM(20))
        self.model.add(Dense(5))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(1))

        self.model.compile(
              loss=self.config['model']['loss'],
              optimizer=optimizers.Adam(self.config['model']['learning_rate']))

    def train(self, features, labels):
        """
        Trains the model on the provided data and save logs.
        :param features: Data matrix of features
        :param labels: Data matrix of labels
        """
        features, labels = utils.data_to_time_steps(features, labels, self.config['time_steps'])
        self.model.fit(
              features, labels, epochs=self.config['model']['epochs'],
              batch_size=self.config['model']['batch_size'], verbose=self.config['model']['verbose'],
              callbacks=[TensorBoard(log_dir=self.config['model']['tensorboard_dir'])],
              validation_split=self.config['validation_split'], shuffle=True)
