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
from keras.optimizers import Adam
from keras.layers import PReLU
from keras.initializers import Constant
from predictCO2.models.nn_template import NN_Template
from tensorflow.keras import layers, models, optimizers, backend
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from kerastuner import HyperModel
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import fbeta_score, make_scorer

tensorflow.get_logger().setLevel('INFO')
from keras.wrappers.scikit_learn import KerasRegressor


class CNN(NN_Template):

    def __init__(self, config, num_features, num_steps, nn1=32,nn2=64,dn1=50,dn2=100,dp=0.1,lr=0.01,decay=1e-5):
        """
        :param config: Configuration file containing parameters
        """
        super(CNN, self).__init__(config)
        self.nn1 = nn1
        self.nn2=nn2
        self.lr=lr
        self.decay=decay
        self.dp=dp
        self.dn1=dn1
        self.dn2=dn2
        self.n_feats = num_features
        self.n_stp = num_steps
        self.build_model()
        self.config = config
        self.n_layer = nn1
        self.prediction_tolerance = 20e-2

    #make sure to pass on all the hyperparameters as your function parameter otherwise RandomSearchCv raises an error.
    def build_model(self,nn1=32,nn2=64,lr=0.01,dp=0.1,decay=1e-4,dn1=50,dn2=100):
        """
        Builds the model as specified in the model configuration file.
        """

        opt = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=self.decay)
        model = models.Sequential()
        model.add(Conv1D(filters=self.nn1, kernel_size=3, padding="same", input_shape=(self.n_stp, self.n_feats)))
        model.add(PReLU(alpha_initializer=Constant(value=0.20)))
        model.add(Conv1D(filters=self.nn2, kernel_size=2, padding="same"))
        model.add(MaxPool1D(pool_size=1))
        model.add(PReLU(alpha_initializer=Constant(value=0.20)))

        model.add(Dropout(self.dp))
        model.add(Flatten())
        model.add(Dense(self.dn1))
        model.add(PReLU(alpha_initializer=Constant(value=0.20)))
        model.add(Dense(self.dn2))
        model.add(PReLU(alpha_initializer=Constant(value=0.20)))
        model.add(Dense(1))
        model.add(Activation('relu'))

        model.compile(loss="mse",
                      optimizer=opt,
                      metrics=[self.soft_acc])

        return model

    def tuning(self, param_grid):
        model = KerasRegressor(build_fn=self.build_model, epochs=10, verbose=1)
        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=KFold(3), n_iter=10)

        return grid

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
            score= backend.mean(backend.abs(y_true - y_pred) <= self.prediction_tolerance)
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
