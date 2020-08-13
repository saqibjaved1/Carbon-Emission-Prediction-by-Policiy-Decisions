"""
Created by: Tapan Sharma
Date: 04/08/20
"""

import tensorflow
from tensorflow.python.keras.callbacks import TensorBoard

from predictCO2.models.nn_template import NN_Template
from tensorflow.keras import layers, models, optimizers, backend, utils

tensorflow.get_logger().setLevel('INFO')


class DeepLearningModel(NN_Template):

    def __init__(self, config, num_features, num_outputs):
        """
        Initializer for CNN.
        :param config: Configuration file containing parameters
        """
        super(DeepLearningModel, self).__init__(config)
        self.n_feats = num_features
        self.n_ops = num_outputs
        self.prediction_tolerance = self.config['model']['prediction_tolerance']
        self.build_model()

    def build_model(self):
        """
        Builds the model as specified in the model configuration file.
        """
        self.model = models.Sequential()
        for layer in self.config['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            filters = layer['filters'] if 'filters' in layer else None
            kernel_size = layer['kernel_size'] if 'kernel_size' in layer else None
            leaky_alpha = layer['leak_factor'] if 'leak_factor' in layer else None
            pool_size = layer['pool_size'] if 'pool_size' in layer else None

            input_dim = self.n_feats

            if layer['type'] == 'dense':
                self.model.add(layers.Dense(neurons, activation=activation))
            if layer['type'] == 'flatten':
                self.model.add(layers.Flatten())
            if layer['type'] == 'lstm':
                self.model.add(
                    layers.LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'conv1d':
                self.model.add(layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                             input_shape=(input_timesteps, input_dim)))
            if layer['type'] == 'leakyrelu':
                self.model.add(layers.LeakyReLU(alpha=leaky_alpha))
            if layer['type'] == 'max_pool':
                self.model.add(layers.MaxPooling1D(pool_size=pool_size))
            if layer['type'] == 'dropout':
                self.model.add(layers.Dropout(dropout_rate))

        self.model.compile(loss=self.config['model']['loss'],
                           optimizer=optimizers.Adam(self.config['model']['learning_rate']),
                           metrics=[self.soft_acc, "mae"])

    def train_with_validation_provided(self, features, labels, val_features, val_labels):
        """
        Trains the model on the provided data and save logs.
        :param features: Data matrix of features
        :param labels: Data matrix of labels
        :return hist: History of training
        """
        hist = self.model.fit(
            features, labels, batch_size=self.config['training']['batch_size'],
            epochs=self.config['training']['epochs'],
            validation_data=(val_features, val_labels),
            validation_freq=self.config['training']['validation_frequency'],
            callbacks=[TensorBoard(log_dir=self.config['model']['tensorboard_dir'])])
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
        return backend.mean(backend.abs(y_true - y_pred) <= self.prediction_tolerance)

    def plot_and_save_model(self, path_to_file):
        utils.plot_model(self.model, to_file=path_to_file, show_shapes=True)