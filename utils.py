import csv
import os
import posixpath


def load_all_params(params_groups=['positive-electrode', 'negative-electrode', 'solid-electrolyte']):
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
    print(params_group)
    with open(params_group + ".csv", "r") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if row['Name [units]'].startswith("#") or row['Name [units]'] == '':
                continue
            params[row['Name [units]']] = row['Value']
    
    return params


def process_params_value(value):
    """"""
    return