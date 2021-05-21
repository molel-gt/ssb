import csv
import os


def get_params(params_group):
    """"""
    return


def read_params_file(params_group):
    """"""
    with open(os.path.join(params_group, ".csv")) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if row['Name'].startswith("#") or row['Name'] == '':
                continue
            yield row


def process_params_value(value):
    """"""
    return
