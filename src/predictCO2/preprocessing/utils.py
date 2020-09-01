"""Created by: Tapan Sharma
Date: 20/06/20
"""
import Globals
import pandas as pd
import yaml
import numpy as np


def conform_date(date_str):
    """
    Method conforms the date in Global Carbon project to match with the date of OxCGRT i.e. "YYYYMMDD" e.g. 20200101
    :param date_str: Date string from Global Carbon project (of form "DD/MM/YYYY")
    :return: date in form "YYYYMMDD"
    """
    if isinstance(date_str, pd.Timestamp):
        return date_str.strftime(('%Y%m%d'))
    else:
        date_parts = None

        if '/' in date_str:
            date_parts = date_str.split('/')
        if '-' in date_str:
            date_parts = date_str.split('-')
        if '.' in date_str:
            date_parts = date_str.split('.')

        if date_parts:
            year = date_parts[2]
            month = date_parts[1]
            day = date_parts[0]
            if len(day) == 1:
                day = '0' + day
            if len(month) == 1:
                month = '0' + month
            conformed_date = year + month + day
            return conformed_date
        else:
            raise ValueError("Failed to conform date in YYYYMMDD format for received: {}".format(date_str))


def load_cfg_file(cfg_name):
    cfg_prefix = Globals.ROOT_DIR + '/cfg/'
    cfg_rel_path = cfg_prefix + "/" + cfg_name
    cfg = None
    with open(r'{}'.format(cfg_rel_path)) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    return cfg


def data_to_time_steps(features, labels, n_input):
    """
    convert data into multi-steps inputs and outputs
    :param features: Data matrix for features
    :param labels: Data matrix for labels
    :param n_input: Number of previous inputs to be considered
    :return: Modified features and labels matrix
    """
    x, y = list(), list()
    in_start = 0
    # iterate through the whole data
    for _ in range(len(features)):
        # define the end of the input sequence
        in_end = in_start + n_input
        # ensure we have enough data for this instance
        if in_end < len(features):
            x.append(features[in_start:in_end])
            y.append(labels[in_end, 0])
        # move along one time step
        in_start += 1
    return np.array(x), np.array(y)


def data_sequence_generator(features, labels, n_steps):
    """
    convert data into overlapping sequences as specified by n_seq
    :param features: Data matrix for features
    :param labels: Data matrix for labels
    :param n_steps: Number of previous inputs to be considered
    :return: Modified features and labels matrix
    """
    x, y = list(), list()
    num_samples = features.shape[0]
    start = 0
    for _ in range(num_samples):
        end = start + n_steps
        if end < num_samples:
            x.append(features.iloc[start:end, :].to_numpy())
            if labels is not None:
                y.append(labels.iloc[end, :].to_numpy())
        start += 1
    return np.array(x), np.array(y)


def time_series_data_generator(features, labels, n_steps):
    """
    convert data into overlapping sequences as specified by n_seq
    :param features: Data matrix for features
    :param labels: Data matrix for labels
    :param n_steps: Number of previous inputs to be considered
    :return: Modified features and labels matrix
    """
    x, y = list(), list()
    feat_row = list(features.index.values)
    start = 0
    for _ in range(len(feat_row)):
        end = start + n_steps
        if end < len(feat_row):
            x.append(features.iloc[start:end, :].to_numpy())
            y.append(labels.iloc[end, 0])
        start += 1
    return np.array(x), np.array(y)


def generate_time_series_df(features, labels, n_steps):
    """
    Concatenates input features with CO2 reductions for the last n time steps
    :param features: Feature or input matrix
    :param labels: Labels or output data
    :param n_steps: number of time steps considered
    :return: Input data frame containing input features and labels for last n steps
    """
    num_samples = features.shape[0]
    x = []
    for i in range(num_samples):
        if i >= n_steps:
            labels_prev_steps = labels.iloc[i - n_steps:i, :].to_numpy()
            x.append(labels_prev_steps)
    df1 = pd.DataFrame(np.squeeze(x))
    df1.set_index(features.index[n_steps:], inplace=True)
    in_data = pd.concat([features.iloc[n_steps:, :], df1], axis=1, ignore_index=True)
    out_data = None
    if labels is not None:
        out_data = labels.iloc[n_steps:, :]
    return in_data, out_data
