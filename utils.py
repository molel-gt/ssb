import os


def make_dir_if_missing(f_path):
    """"""
    os.makedirs(f_path, exist_ok=True)