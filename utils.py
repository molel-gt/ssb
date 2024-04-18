import os


def make_dir_if_missing(f_path):
    """"""
    os.makedirs(f_path, exist_ok=True)


def print_dict(dict, padding=20):
    for k, v in dict.items():
        print(k.ljust(padding, ' '), ": ", str(v))