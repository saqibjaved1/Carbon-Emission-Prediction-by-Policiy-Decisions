"""
Author: Sreetama Sarkar
Date: 7/25/2020
"""
import os
import json
from tensorflow.keras.utils import plot_model
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from predictCO2.models.mixed_nn_model import nn_model
from predictCO2.preprocessing import utils
from predictCO2.preprocessing.generate_data import CountryPolicyCarbonData, PolicyCategory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns

with open('cfg/densenn_config.json') as f:
    training_config = json.load(f)

countries = training_config['countries']
train_features = pd.DataFrame()
train_labels = pd.DataFrame()
test_features = pd.DataFrame()
test_labels = pd.DataFrame()
norm_data = training_config['training']['normalize']

# Collect data
for country in countries:
    countryPolicyCarbonData = CountryPolicyCarbonData('training_data.yaml', country, include_flags=False,
                                                      policy_category=PolicyCategory.SOCIAL_INDICATORS,
                                                      normalize=0)
    train_x, train_y, test_x, test_y = countryPolicyCarbonData.split_train_test(fill_nan=False)
    train_features = train_features.append(train_x)
    test_features = test_features.append(test_x)
    train_labels = train_labels.append(train_y)
    test_labels = test_labels.append(test_y)

print(train_features.shape)
print(train_labels.shape)
print(test_features.shape)
print(test_labels.shape)

# Convert to time-series data
features, labels = utils.time_series_data_generator(train_features, train_labels, training_config['time_steps'])
test_f, test_l = utils.time_series_data_generator(test_features, test_labels, training_config['time_steps'])
# Split train data into train and validation sets
train_f, val_f, train_l, val_l = train_test_split(features, labels, test_size=0.2)

# # Visualize feature label correlations
# aug_data = pd.concat([train_f, train_y], axis=1, ignore_index=True)
# sns.pairplot(aug_data)
# plt.show()

# Train model
_, _, n_features = train_f.shape
nn = nn_model(training_config, num_features=n_features, num_outputs=1)
print(nn.model.summary())
h = nn.train_with_validation_provided(train_f, train_l, val_f, val_l)
# conv.model.load_weights("best_epoch.h5")
train_loss = h.history['loss']
val_loss = h.history['val_loss']
#
# Plot training loss
fig1, ax1 = plt.subplots()
ax1.plot(range(len(train_loss)), train_loss, label='Training Loss')
ax1.plot(range(len(val_loss)), val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.show()


# # Plot training labels vs predicted labels
# y_pred = nn.model.predict(features)
# fig2, ax2 = plt.subplots()
# ax2.plot(range(len(labels)), labels, label='True Labels')
# ax2.plot(range(len(labels)), y_pred, label='Predicted Labels')
# plt.xlabel('Time')
# plt.ylabel('y values')
# plt.legend(loc='upper right')
# plt.title('Training Data')
# plt.show()
#
# ## Plot test labels vs predicted labels
# y_pred = nn.model.predict(val_f)
# fig2, ax2 = plt.subplots()
# ax2.plot(range(len(val_l)), val_l, label='True Labels')
# ax2.plot(range(len(val_l)), y_pred, label='Predicted Labels')
# plt.xlabel('Time')
# plt.ylabel('y values')
# plt.legend(loc='upper right')
# plt.title('Validation Data')
# plt.show()

# Prediction
print("Test Set Evaluation:\n")
model_eval = nn.model.evaluate(test_f, test_l)
print("\n\nMSE: {}\nMAE: {}\nR2 Score: {}\nSoft Accuracy: {}".format(model_eval[0], model_eval[1], model_eval[2], model_eval[3]))

# plot_model(nn.model)
nn.save("DNN_Sreetama")
# # Save best model and log metrics
# mse = nn.model.evaluate(val_f, val_l)[0]
# metric_file = training_config["model"]["checkpoint_path"]+"Best_Metrics_" + training_config["country"] + ".txt"
# try:
#     f = open(metric_file)
#     old_mse = f.readline().split(': ')[1]
#     if mse < int(old_mse):
#         f.write("mse: " + str(mse))
#         filename = "best_models/bestnn_"+training_config["country"]+".h5"
#         filepath = os.path.join(training_config["model"]["checkpoint_path"], filename)
#         nn.model.save(filepath)
# except:
#     f = open(metric_file, 'w')
#     f.write("mse: "+str(mse))
#     filename = "best_models/bestnn_" + training_config["country"] + ".h5"
#     filepath = os.path.join(training_config["model"]["checkpoint_path"], filename)
#     nn.model.save(filepath)
#     f.close()


