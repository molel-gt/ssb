#! /usr/bin/env python3

import os

import argparse
import matplotlib.pyplot as plt
import numpy as np

def build_graph(array_chunk):
    return


def chuck_array(data, chuck_max_size):
    return


def filter_interior_points(data):
    """
    Masks locations where the voxel has 8 neighbors and each the 8 neighbors
    has 8 neighbors
    """
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='computes specific area')
    parser.add_argument('--working_dir', help='bmp files directory', required=True)

    args = parser.parse_args()
    working_dir = args.working_dir
    im_files = [os.path.join(working_dir, f) for f in os.listdir(working_dir) if f.endswith(".bmp")]
    n_files = len(im_files)