"""
Author: Saqib Javed
Date: 5/6/2020
"""

import sys
import json
from predictCO2.models.lstm import Lstm
from predictCO2.preprocessing import generate_data
from sklearn import preprocessing


def main():
    if len(sys.argv) != 2:
        print("Cannot find Config file!")
        sys.exit(1)
    else:
        config_path = sys.argv[1:][0]

    with open(config_path, 'r') as jsonfile:
        config = json.load(jsonfile)

    data = generate_data.CountryPolicyCarbonData("training_data.yaml", config["country"])
    aug_data = data.get_augmented_data(generate_data.DataType.PANDAS_DF)
    aug_data = aug_data.reset_index()
    aug_data = aug_data.drop(["index"], axis=1)
    aug_data = aug_data.drop([1, 3, 5, 7, 9, 11, 13])
    features = preprocessing.scale(aug_data[:8].values.T)
    labels = aug_data[19:].values.T

    model = Lstm(config)
    model.train(features, labels)
    model.save("Checkpoint")


if __name__ == "__main__":
    main()
