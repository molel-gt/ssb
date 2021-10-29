#!/usr/bin/env python3

import os

import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_images_to_logical_array(files_list, file_shape, limits=None):
    """"""
    n_start = 0
    n_end = file_shape[0]
    if limits is not None:
        n_start, n_end = limits
    n_files = len(files_list)
    new_shape = [n_end - n_start, n_end - n_start, n_end - n_start]
    data = np.zeros(new_shape, dtype=bool)
    for i_x, img_file in enumerate(files_list):
        if i_x < n_start or i_x > n_end:
            continue
        img_data = plt.imread(img_file)
        img_data = img_data / 255
        img_data = img_data[n_start:n_end, n_start:n_end]
        data[i_x-n_start-1, :, :] = img_data

    return data


def compute_boundary_markers(local_pos, grid_shape):
    """"""
    # TODO: determine whether the position is at the faces of the box
    x, y, z = local_pos
    if x == 0:
        return 1
    elif y == 0:
        return 4
    elif z == 0:
        return 5
    elif x == grid_shape[0] - 1:
        return 3
    elif y == grid_shape[1] - 1:
        return 2
    elif z == grid_shape[2] - 1:
        return 6
    return 0


def create_nodes(data, **kwargs):
    """"""
    n_nodes = np.sum(data)
    nodes = np.zeros([n_nodes, 4])
    count = 0
    for idx, point in np.ndenumerate(data):
        if point:
            boundary_marker = compute_boundary_markers(idx, data.shape)
            nodes[count, :] = list(idx) + [boundary_marker]
            count += 1
    return nodes


def write_node_to_file(nodes, node_file_path):
    """"""
    count, _ = nodes.shape
    meta_header = "# Node count, 3 dim, no attribute, no boundary marker"
    header_data = [str(count), '3', '0', '1']
    entries_header = "# Node index, node coordinates, boundary marker"
    with open(node_file_path, "w") as fp:
        fp.write(meta_header + '\n')
        fp.write(' '.join(header_data) + '\n')
        fp.write(entries_header + '\n')
        for idx in range(count):
            entry = [idx] + list(nodes[idx, :])
            fp.write(' '.join([str(v) for v in entry]) + '\n')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='creates node file for meshing..')
    parser.add_argument('--working_dir', help='bmp_files parent directory', required=True)
    parser.add_argument('--file_shape', help='shape of image data array', required=True,
                        type=lambda s: [int(item) for item in s.split(',')])

    args = parser.parse_args()
    files_dir = os.path.join(args.working_dir, 'bmp_files')
    file_shape = args.file_shape
    files_list = sorted([os.path.join(files_dir, f) for f in os.listdir(files_dir)
                  if f.endswith(".bmp")])
    files_list = [f for i, f in enumerate(files_list) if i >= 2 and i < 92]
    print("loading image files to logical array..")
    image_data = load_images_to_logical_array(files_list, file_shape, limits=[10, 81])
    node_file_path = os.path.join(args.working_dir, 'porous-solid.node')
    nodes = create_nodes(image_data)
    write_node_to_file(nodes, node_file_path)
