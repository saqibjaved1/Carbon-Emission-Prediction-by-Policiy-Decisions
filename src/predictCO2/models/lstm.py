"""
Author: Saqib Javed
Date: 01/7/2020
"""
from src.predictCO2.models import nn_template
from src.predictCO2.preprocessing import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Conv2D,Activation,Dropout,Flatten
from tensorflow.keras import optimizers,backend
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from tensorflow.keras import backend as K

class Lstm(nn_template.NN_Template):
    def __init__(self, config):
        super(Lstm, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(20,input_shape=(self.config['time_steps'],8),return_sequences=True))
        self.model.add(Dropout(0.4))
        self.model.add(LSTM(20))        
        self.model.add(Dense(5))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(1))
        
        self.model.compile(
              loss=self.config['model']['loss'],
              optimizer=optimizers.Adam(self.config['model']['learning_rate']),
              metrics=['mae', self.soft_acc])
    
    def train(self,features,labels,val_features,val_labels):
        #features, labels = utils.data_to_time_steps(features,labels,self.config['time_steps'])
        #val_features, val_labels = utils.data_to_time_steps(features,labels,self.config['time_steps'])
        callbacks =[
                    EarlyStopping(monitor='val_loss', patience=5, mode='min'),
                    ModelCheckpoint(self.config["model"]["checkpoint_path"], save_best_only=True, monitor='val_loss', mode='min'),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, epsilon=1e-4, mode='min',min_lr=0.0001),
                    TensorBoard(log_dir=self.config['model']['tensorboard_dir'])
                    ]
        return self.model.fit(
              features,labels,epochs=self.config['model']['epochs'],
              batch_size=self.config['model']['batch_size'],verbose = self.config['model']['verbose'],
              callbacks = callbacks,
              validation_data=(val_features, val_labels))

    def soft_acc(self, y_true, y_pred):
        """
        Evaluates soft accuracy by comparing ground truth label with the predicted label within some tolerance level.
        :param y_true: Ground truth
        :param y_pred: Predictions
        :return: normalized accuracy score
        """
        return backend.mean(backend.abs(backend.round(y_true) - backend.round(y_pred)) <= self.config['prediction_tolerance'])

    def r2_keras(self, y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - SS_res / (SS_tot + K.epsilon())
