#!/usr/bin/env python3

import os

import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_images_to_logical_array(files_list, grid_info):
    """
    grid_sizes: Lx.Ly.Lz
    """
    Lx, Ly, Lz = grid_sizes.split('.')
    data = np.zeros([int(Lx), int(Ly), int(Lz)], dtype=bool)
    for i_x, img_file in enumerate(files_list):
        img_data = plt.imread(img_file)
        img_data = img_data / 255
        data[i_x, :, :] = img_data
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
    n_nodes = int(np.sum(data))
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
    parser.add_argument('--working_dir', help='bmp files parent directory', required=True)
    parser.add_argument('--img_sub_dir', help='bmp files parent directory', required=True)
    parser.add_argument('--grid_info', help='grid_size, start_pos, end_pos', required=True)

    args = parser.parse_args()
    files_dir = os.path.join(args.working_dir, args.img_sub_dir)
    grid_info = args.grid_info
    grid_sizes = grid_info
    files_list = sorted([os.path.join(files_dir, f) for f in os.listdir(files_dir)
                  if f.endswith(".bmp")])
    image_data = load_images_to_logical_array(files_list, grid_info)
    print("porosity: ", np.average(image_data))
    meshes_dir = os.path.join(args.working_dir, 'mesh')
    node_file_path = os.path.join(meshes_dir, '{}.node'.format(grid_sizes))
    nodes = create_nodes(image_data)
    write_node_to_file(nodes, node_file_path)
