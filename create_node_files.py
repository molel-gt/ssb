#!/usr/bin/env python3

import os

import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_images_to_logical_array(files_list, file_shape):
    """"""
    n_files = len(files_list)
    data_shape = [n_files] + list(file_shape)
    data = np.zeros(data_shape, dtype=bool)
    for i_x, img_file in enumerate(files_list):
        img_data = plt.imread(img_file)
        data[i_x, :, :] = img_data[:]

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
    parser = argparse.ArgumentParser(description='creates node file for meshing..')
    parser.add_argument('--files_dir', help='image files directory', required=True)
    parser.add_argument('--file_shape', help='shape of image data array', required=True,
                        type=lambda s: [int(item) for item in s.split(',')])

    args = parser.parse_args()
    files_dir = args.files_dir
    file_shape = args.file_shape
    files_list = [os.path.join(files_dir, f) for f in os.listdir(files_dir)
                  if f.endswith(".bmp")]
    print("loading image files to logical array..")
    image_data = load_images_to_logical_array(files_list, file_shape)
