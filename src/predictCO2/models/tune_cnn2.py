"""
Created by: Tapan Sharma
Date: 03/09/20
"""
from kerastuner import HyperModel, RandomSearch, Objective, Hyperband
from tensorflow.keras import layers, models, optimizers, backend, utils


class CNN2(HyperModel):
    def __init__(self, config, num_features, num_outputs):
        """
        Initializer for CNN.
        :param config: Configuration file containing parameters
        """
        self.n_feats = num_features
        self.n_ops = num_outputs
        self.config = config
        self.prediction_tolerance = self.config['model']['prediction_tolerance']

    def build_model(self, hp):
        """
        Builds the model as specified in the model configuration file.
        """
        self.model = models.Sequential()
        self.model.add(layers.Conv1D(hp.Int('conv1d_1', 16, 128, 4),
                                     kernel_size=2,
                                     padding="same",
                                     input_shape=(7, self.n_feats)))
        self.model.add(layers.LeakyReLU(alpha=0.001))
        self.model.add(layers.MaxPooling1D(pool_size=2))
        self.model.add(layers.Conv1D(hp.Int('conv1d_2', 16, 128, 4),
                                     kernel_size=2,
                                     padding="same"))
        self.model.add(layers.LeakyReLU(alpha=0.001))
        self.model.add(layers.MaxPooling1D(pool_size=2))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(hp.Int('dense_1', 4, 100, 4)))
        self.model.add(layers.LeakyReLU(alpha=0.001))
        self.model.add(layers.Dropout(hp.Float('dropout', 0.01, 0.5, 0.01)))
        self.model.add(layers.Dense(1))
        self.model.compile(loss="huber",
                                  optimizer=optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                                  metrics=[self.soft_acc, "mae"])
        return self.model

    def soft_acc(self, y_true, y_pred):
        """
        Evaluates soft accuracy by comparing ground truth label with the predicted label within some tolerance level.
        :param y_true: Ground truth
        :param y_pred: Predictions
        :return: normalized accuracy score
        """
        try:
            return backend.mean(backend.equal(backend.round(y_true), backend.round(y_pred)))
        except Exception:
            pass

    def tuning(self, method="random"):
        if method == "random":
            return RandomSearch(self.build_model, objective='mae', max_trials=5, executions_per_trial=2,
                                directory='./tuning', project_name='CNN2')
        if method == "hyperband":
            return Hyperband(self.build_model, objective='mae', max_epochs=20, factor=2, directory='./tuning',
                             project_name='CNN2_hyperband_huber')
