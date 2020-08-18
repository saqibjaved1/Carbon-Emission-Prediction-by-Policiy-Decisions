"""
Author: Sreetama Sarkar
Date: 7/25/2020
"""
import os
import tensorflow
from tensorflow.python.keras.callbacks import TensorBoard

from predictCO2.models.nn_template import NN_Template
from tensorflow.keras import layers, models, optimizers, backend
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K


class nn_model(NN_Template):

    def __init__(self, config, num_features, num_outputs):
        """
        Initializer for CONV1D model
        :param config: Configuration file containing parameters
        """
        super(nn_model, self).__init__(config)
        self.n_feats = num_features
        self.n_steps = self.config['time_steps']
        self.n_ops = num_outputs
        self.prediction_tolerance = self.config['model']['prediction_tolerance']
        self.build_model()

    def build_model(self):
        """
        Builds the model as specified in the model configuration file.
        """
        self.model = models.Sequential()
        for layer in self.config['model']['layers']:

            if layer['type'] == 'conv1D':
                if layer['index'] == 0:
                    self.model.add(layers.Conv1D(filters=layer['filters'], kernel_size=layer['kernel_size'],
                                                 activation=layer['activation'],
                                                 input_shape=(self.n_steps, self.n_feats)))
                else:
                    self.model.add(layers.Conv1D(filters=layer['filters'], kernel_size=layer['kernel_size'],
                                                 activation=layer['activation']))

            if layer['type'] == 'dense':
                if layer['index'] == 0:
                    self.model.add(layers.Dense(layer['units'], activation=layer['activation'], input_shape=(self.n_steps, self.n_feats)))
                else:
                    self.model.add(layers.Dense(layer['units'], activation=layer['activation']))

            if layer['type'] == 'flatten':
                if layer['index'] == 0:
                    self.model.add(layers.Flatten(input_shape=(self.n_steps, self.n_feats)))
                else:
                    self.model.add(layers.Flatten())

            if layer['type'] == 'dropout':
                self.model.add(layers.Dropout(layer['rate']))

        self.model.compile(loss=self.config['model']['loss'],
                           optimizer=optimizers.Adam(self.config['model']['learning_rate']),
                           metrics=['mae', self.r2_keras, self.soft_acc])

    def train_with_validation_provided(self, features, labels, val_features, val_labels):
        """
        Trains the model on the provided data and save logs.
        :param features: Data matrix of features
        :param labels: Data matrix of labels
        :return hist: History of training
        """

        checkpoint_path = self.config["model"]["checkpoint_path"]
        # filename = "best_epoch_"+self.config["country"]+".h5"
        filename = "best_epoch.h5"
        checkpoint_dir = os.path.join(checkpoint_path, filename)

        # print(os.path.isdir(checkpoint_path))
        # Create a callback that saves the model's weights
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint(checkpoint_dir, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [es, mc]
        # This may generate warnings related to saving the state of the optimizer.
        # These warnings (and similar warnings throughout this notebook)
        # are in place to discourage outdated usage, and can be ignored.
        hist = self.model.fit(
            features, labels, batch_size=self.config['training']['batch_size'],
            epochs=self.config['training']['epochs'],
            validation_data=(val_features, val_labels), callbacks = callbacks_list)
            # callbacks = [TensorBoard(log_dir=self.config['model']['tensorboard_dir'], profile_batch=100000000)]) #For Windows10
        return hist

    def train(self, features, labels):
        pass

    def soft_acc(self, y_true, y_pred):
        """
        Evaluates soft accuracy by comparing ground truth label with the predicted label within some tolerance level.
        :param y_true: Ground truth
        :param y_pred: Predictions
        :return: normalized accuracy score
        """
        return backend.mean(backend.abs(backend.round(y_true) - backend.round(y_pred)) <= self.prediction_tolerance)

    def r2_keras(self, y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - SS_res / (SS_tot + K.epsilon())