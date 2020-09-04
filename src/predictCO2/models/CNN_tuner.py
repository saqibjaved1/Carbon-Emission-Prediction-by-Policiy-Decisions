"""
Created by : Subarnaduti Paul
Date: 27/08/20
"""
import tensorflow
from tensorflow.python.keras.callbacks import TensorBoard
from keras.models import Sequential, load_model
import os
from keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Dropout, Activation, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import  make_scorer
from keras.optimizers import Adam
from keras.layers import PReLU
from keras.initializers import Constant
from predictCO2.models.nn_template import NN_Template
from tensorflow.keras import layers, models, optimizers, backend
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from kerastuner import HyperModel

tensorflow.get_logger().setLevel('INFO')


class CNN(HyperModel):

    def __init__(self, config, num_features, num_steps):
        """
        :param config: Configuration file containing parameters
        """
        # super(CNN, self).__init__(config)

        self.n_feats = num_features
        self.n_stp = num_steps
        # self.build_model()
        self.config = config

        self.prediction_tolerance = 20e-2

    #Include hp as a parameter to model building function to call Hperparameter method of Keras Tuner
    def build_model(self, hp):
        """
        Builds the model as specified in the model configuration file.
        """

        opt = Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]), beta_1=0.9, beta_2=0.999)
        model = models.Sequential()
        model.add(Conv1D(hp.Int('Int_layer', 32, 96, 16), kernel_size=3, padding="same",
                         input_shape=(self.n_stp, self.n_feats)))
        model.add(PReLU(alpha_initializer=Constant(value=0.20)))
        model.add(Conv1D(hp.Int('Int_layer', 64, 96, 16), kernel_size=2, padding="same"))
        model.add(MaxPool1D(pool_size=1))
        model.add(PReLU(alpha_initializer=Constant(value=0.20)))

        model.add(Dropout(hp.Choice('dropout', values=[0.1, 0.2])))
        model.add(Flatten())
        model.add(Dense(hp.Int('Dense_layer1', 50, 100, 15)))
        model.add(PReLU(alpha_initializer=Constant(value=0.20)))
        model.add(Dense(hp.Int('Dense_layer1', 90, 120, 10)))
        model.add(PReLU(alpha_initializer=Constant(value=0.20)))
        model.add(Dense(1))
        model.add(Activation('tanh'))

        model.compile(loss="mse",
                      optimizer=opt,
                      metrics=[self.soft_acc])

        return model
    #RandomSearch function of Keras tuner
    def tuning(self):
        tuner = RandomSearch(self.build_model, objective='val_accuracy', max_trials=2, executions_per_trial=2,
                             directory='./cfg', project_name='Tuner')

        return tuner

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
        try:
            score = backend.mean(backend.abs(y_true - y_pred) <= self.prediction_tolerance)
        except Exception:
            pass
        return score

    def save(self, name):
        """
        Saves the model checkpoint to the path specified by the argument
        """
        if self.model is None:
            raise Exception("Build the model first!")

        if os.path.isdir(self.config["model"]["checkpoint_path"]) is False:
            os.makedirs(self.config["model"]["checkpoint_path"])

        print("Saving model...")
        self.model.save_weights(self.config["model"]["checkpoint_path"] + name)
        print("Model saved!")

    def load(self):
        """
        Loads the model checkpoint from the path specified by the argument
        """
        if self.model is None:
            raise Exception("Build the model first.")

        print("Loading model checkpoint {} ...\n".format(self.config["model"]["restore_model"]))
        self.model.load_weights(self.config["model"]["restore_model"])
        print("Model loaded!")
