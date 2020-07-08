"""
Author: Sreetama Sarkar
Date: 7/8/2020
"""

import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

import Globals
import predictCO2.preprocessing.utils as utils
from src.predictCO2.preprocessing import generate_data
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter


def plot_xy(x, y, format, xlabel, ylabel, title, fir_dir, fig_name):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax.plot(x, y, format)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.xlim(-1, 5)
    plt.savefig(fir_dir + '/' + fig_name)
    plt.show()


def plot_stringency_index(df, country_name, config_file):
    training_cfg = utils.load_cfg_file(config_file)
    policy_csv = Globals.ROOT_DIR + "/" + training_cfg['features']
    policy_data = pd.read_excel(policy_csv, sheet_name=country_name)
    str_index = policy_data['StringencyIndexForDisplay'].tolist()
    print(str_index)

    # Labels
    co2 = df.iloc[-1, :].tolist()

    min_len = len(str_index) if (len(co2) > len(str_index)) else len(co2)
    str_index = str_index[:min_len]
    co2 = co2[:min_len]

    # Plot configurations
    x_label = 'Government Response Stringency Index ((0 to 100, 100 = strictest))'
    y_label = 'Reduction in Co2 Emissions'
    title = 'Stringency Index vs CO2 Reduction ' + '(' + country_name + ')'
    fig_dir = './plots'
    fig_name = 'stringency_vs_co2_' + country_name + '.png'
    format = 'go'
    plot_xy(str_index, co2, format, x_label, y_label, title, fig_dir, fig_name)


def plot_features_vs_labels(df, policy, country_name):
    # feature_dict
    feature_dict = {'c1': {'data': df.iloc[0, :], 'name': generate_data.PolicyData.C1},
                    'c2': {'data': df.iloc[2, :], 'name': generate_data.PolicyData.C2},
                    'c3': {'data': df.iloc[4, :], 'name': generate_data.PolicyData.C3},
                    'c4': {'data': df.iloc[6, :], 'name': generate_data.PolicyData.C4},
                    'c5': {'data': df.iloc[8, :], 'name': generate_data.PolicyData.C5},
                    'c6': {'data': df.iloc[10, :], 'name': generate_data.PolicyData.C6},
                    'c7': {'data': df.iloc[12, :], 'name': generate_data.PolicyData.C7},
                    'c8': {'data': df.iloc[14, :], 'name': generate_data.PolicyData.C8}}

    feature = feature_dict[policy]['data'].tolist()

    # Labels
    co2 = df.iloc[-1, :].tolist()
    # co2 = [round(float(elem), 2) for elem in co2]

    # Plot configurations
    x_label = feature_dict[policy]['name']
    y_label = 'Reduction in Co2 Emissions'
    title = 'Policy vs CO2 Reduction ' + '(' + country_name + ')'
    fig_dir = './plots'
    fig_name = policy + '_vs_co2_' + country_name + '.png'
    format = 'ro'
    plot_xy(feature, co2, format, x_label, y_label, title, fig_dir, fig_name)


def main():
    if len(sys.argv) != 2:
        print("Cannot find Config file!")
        sys.exit(1)
    else:
        config_path = sys.argv[1:][0]

    with open(config_path, 'r') as jsonfile:
        config = json.load(jsonfile)

    config_file = config["config_file"]
    country_name = config["country"]
    policy = config["policy"]

    data_type = generate_data.DataType.PANDAS_DF
    data = generate_data.CountryPolicyCarbonData(config_file, country_name)
    df = data.get_augmented_data(data_type)
    df.fillna(0, inplace=True)

    plot_features_vs_labels(df, policy, country_name)
    plot_stringency_index(df, country_name, config_file)


if __name__ == "__main__":
    main()


