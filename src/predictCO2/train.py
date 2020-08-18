"""
Author: Saqib Javed
Date: 5/6/2020
"""


import sys,json
sys.path.append("/content")
from predictCO2.models.lstm import Lstm
from predictCO2.preprocessing import generate_data
from sklearn import preprocessing
from src.predictCO2.preprocessing import utils
from keras.utils.vis_utils import plot_model

def main():
  if len(sys.argv) != 2:
        print ("Cannot find Config file!")
        sys.exit(1)
  else:
      config_path = sys.argv[1:][0]
  
  with open(config_path, 'r') as jsonfile:
      config = json.load(jsonfile)

  data= generate_data.CountryPolicyCarbonData("training_data.yaml",config["country"])
  print(config)
  aug_data = data.get_augmented_data(generate_data.DataType.PANDAS_DF)

  model=Lstm(config)

  aug_data = aug_data.reset_index()
  aug_data = aug_data.drop(["index"],axis=1)
  aug_data = aug_data.drop([1,3,5,7,9,11,13])
  features = preprocessing.scale(aug_data[:8].values.T)
  labels = aug_data[19:].values.T
  features, labels = utils.data_to_time_steps(features,labels,3)

  train_X = features[:int(160*0.8)]
  train_Y = labels[:int(160*0.8)]
  val_X = features[int(160*0.8):int(160*0.9)]
  val_Y = labels[int(160*0.8):int(160*0.9)]
  test_X = features[int(160*0.9):]
  test_Y = labels[int(160*0.9):]
  
  history = model.train(train_X,train_Y,val_X,val_Y)
  model.model.evaluate(test_X,test_Y)
  #plot_model(model.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
  model.save("saqib")
if __name__ == "__main__":
    main()  

