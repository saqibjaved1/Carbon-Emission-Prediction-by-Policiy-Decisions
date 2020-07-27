"""
Author: Sreetama Sarkar
Date: 7/25/2020
"""
import os
import json
from tensorflow.keras import models
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from predictCO2.models.mixed_nn_model import nn_model
from predictCO2.preprocessing import utils
from predictCO2.preprocessing.generate_data import CountryPolicyCarbonData
import seaborn as sns

with open('cfg/densenn_config.json') as f:
    training_config = json.load(f)

countryPolicyCarbonData = CountryPolicyCarbonData('training_data.yaml', country=training_config["country"])
train_x, train_y, test_x, test_y = countryPolicyCarbonData.split_train_test(fill_nan=True, validation_percentage=training_config["validation_split"])
#Select only C-Flags as features
train_f = train_x.iloc[:, [0, 2, 4, 6, 8, 10, 12, 14]]
test_f = test_x.iloc[:, [0, 2, 4, 6, 8, 10, 12, 14]]
features, labels = utils.time_series_data_generator(train_f, train_y, training_config['time_steps'])
val_f, val_l = utils.time_series_data_generator(test_f, test_y, training_config['time_steps'])

# Visualize feature label correlations
aug_data = pd.concat([train_f, train_y], axis=1, ignore_index=True)
sns.pairplot(aug_data)
plt.show()

# Train model
_, n_features = train_f.shape
nn = nn_model(training_config, num_features=n_features, num_outputs=1)
print(nn.model.summary())
h = nn.train_with_validation_provided(features, labels, val_f, val_l)
# conv.model.load_weights("best_epoch.h5")
loss = h.history['loss']

# Plot training loss
fig1, ax1 = plt.subplots()
ax1.plot(range(len(loss)), loss, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.show()

# Plot training labels vs predicted labels
y_pred = nn.model.predict(features)
fig2, ax2 = plt.subplots()
ax2.plot(range(len(labels)), labels, label='True Labels')
ax2.plot(range(len(labels)), y_pred, label='Predicted Labels')
plt.xlabel('Time')
plt.ylabel('y values')
plt.legend(loc='upper right')
plt.title('Training Data')
plt.show()

## Plot test labels vs predicted labels
y_pred = nn.model.predict(val_f)
fig2, ax2 = plt.subplots()
ax2.plot(range(len(val_l)), val_l, label='True Labels')
ax2.plot(range(len(val_l)), y_pred, label='Predicted Labels')
plt.xlabel('Time')
plt.ylabel('y values')
plt.legend(loc='upper right')
plt.title('Validation Data')
plt.show()

# Prediction
print("Test Set Evaluation:\n")
model_eval = nn.model.evaluate(val_f, val_l)
print("\n\nMSE: {}\nMAE: {}\nR2 Score: {}\nSoft Accuracy: {}".format(model_eval[0], model_eval[1], model_eval[2], model_eval[3]))

# Save best model and log metrics
mse = nn.model.evaluate(val_f, val_l)[0]
metric_file = training_config["model"]["checkpoint_path"]+"Best_Metrics_" + training_config["country"] + ".txt"
try:
    f = open(metric_file)
    old_mse = f.readline().split(': ')[1]
    if mse < int(old_mse):
        f.write("mse: " + str(mse))
        filename = "best_models/bestnn_"+training_config["country"]+".h5"
        filepath = os.path.join(training_config["model"]["checkpoint_path"], filename)
        nn.model.save(filepath)
except:
    f = open(metric_file, 'w')
    f.write("mse: "+str(mse))
    filename = "best_models/bestnn_" + training_config["country"] + ".h5"
    filepath = os.path.join(training_config["model"]["checkpoint_path"], filename)
    nn.model.save(filepath)
    f.close()


