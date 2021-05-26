import csv
import os
import numpy as np


def load_all_params(params_groups=['separator', 'experiment', 'positive-electrode', 'negative-electrode', 'solid-electrolyte']):
    """"""
    params = {}
    for params_group in params_groups:
        params.update(read_params_file(os.path.join('/home/lesh/ssb', params_group)))

    return params


def get_params(params_group):
    """"""
    return


def read_params_file(params_group):
    """"""
    params = {}
    with open(params_group + ".csv", "r") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if row['Name [units]'].startswith("#") or row['Name [units]'] == '':
                continue
            value = row['Value']
            if value == '':
                value = np.nan
            params[row['Name [units]']] = value
    
    return params


def process_params_value(value):
    """"""
    return