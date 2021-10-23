#!/usr/bin/env python3

import os
import sys

import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_images_to_logical_array(files_list, file_shape):
    """"""
    n_files = len(files_list)
    data_shape = [n_files].append(file_shape)
    data = np.zeros(data_shape, dtype=bool)
    for i_x, img_file in enumerate(files_list):
        data[i_x, :, :] = plt.imread(img_file)

    return data


def compute_boundary_markers(local_pos, grid_shape):
    """"""
    return


def create_nodes(data, **kwargs):
    """"""
    return


def write_node_to_file(nodes):
    """"""
    return


if __name__ == '__main__':
    files_dir = sys.argv[1]
    file_shape = sys.argv[2]
    files_list = [os.path.join(f) for f in os.listdir(files_dir)
                  if f.endswith(".bmp")]
    print("loading image files to logical array..")
    image_data = load_images_to_logical_array(files_list, file_shape)
