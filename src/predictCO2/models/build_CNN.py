"""
Created by : Subarnaduti Paul
Date: 11/08/20
"""
import tensorflow
from tensorflow.python.keras.callbacks import TensorBoard
from keras.models import Sequential, load_model
from keras.optimizers import Adam, Nadam
from keras.initializers import glorot_uniform
from keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Dropout, Activation, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.layers import PReLU
from keras.initializers import Constant
from predictCO2.models.nn_template import NN_Template
from tensorflow.keras import layers, models, optimizers, backend
tensorflow.get_logger().setLevel('INFO')


class CNN(NN_Template):

    def __init__(self, config, num_features, num_steps):
        """
        :param config: Configuration file containing parameters
        """
        super(CNN, self).__init__(config)
        self.n_feats = num_features
        self.n_stp = num_steps
        self.build_model()
        self.prediction_tolerance = 20e-2

    def build_model(self):
        """
        Builds the model as specified in the model configuration file.
        """
        self.model = models.Sequential()
        self.model.add(Conv1D(filters=32, kernel_size=3, padding="same", input_shape=(self.n_stp, self.n_feats)))
        self.model.add(PReLU(alpha_initializer=Constant(value=0.20)))
        self.model.add(Conv1D(filters=64, kernel_size=2, padding="same"))
        self.model.add(MaxPool1D(pool_size=1))
        self.model.add(PReLU(alpha_initializer=Constant(value=0.20)))

        self.model.add(Dropout(0.1))
        self.model.add(Flatten())
        self.model.add(Dense(100))
        self.model.add(PReLU(alpha_initializer=Constant(value=0.20)))
        self.model.add(Dense(50))
        self.model.add(PReLU(alpha_initializer=Constant(value=0.20)))
        self.model.add(Dense(1))
        self.model.add(Activation('tanh'))


        self.model.compile(loss=self.config['model']['loss'],
                           optimizer=optimizers.Adam(self.config['model']['learning_rate']),
                           metrics=[self.soft_acc])
        model = Sequential()

        return model

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
