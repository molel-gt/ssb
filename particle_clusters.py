#! /usr/bin/env python3

import os

import argparse
import numpy as np

from collections import defaultdict
from scipy import linalg

import geometry


def get_neighbors(array_chunk):
    neighbors = np.zeros(array_chunk.shape)

    for idx, val in np.ndenumerate(array_chunk):
        n_neighbors = 0
        if val:
            for counter in range(6):
                new_idx = (idx[0], idx[1], idx[2])
                if counter == 0:
                    # x+1
                    new_idx = (idx[0] + 1, idx[1], idx[2])
                elif counter == 1:
                    # x-1
                    new_idx = (idx[0] - 1, idx[1], idx[2])
                elif counter == 2:
                    # y+1
                    new_idx = (idx[0], idx[1] + 1, idx[2])
                elif counter == 3:
                    # y-1
                    new_idx = (idx[0], idx[1] - 1, idx[2])
                elif counter == 4:
                    # z+1
                    new_idx = (idx[0], idx[1], idx[2] + 1)
                elif counter == 5:
                    # z-1
                    new_idx = (idx[0], idx[1], idx[2] - 1)
                else:
                    raise Exception("Invalid counter")

                try:
                    neigbor_val = array_chunk[tuple(new_idx)]
                except:
                    neigbor_val = 0

                if neigbor_val:
                    n_neighbors += 1
            neighbors[idx] = n_neighbors

    return neighbors


def build_graph(array_chunk):
    """
    :returns: graph
    :rtype: sparse matrix
    """
    n = int(np.sum(array_chunk))
    graph = np.zeros((n, n))
    points = defaultdict(lambda: -1, {})
    valid_points = set([tuple(v) for v in np.argwhere(array_chunk == 1)])
    for idx, value in enumerate(valid_points):
        points[(value[0], value[1], value[2])] = idx

    for idx in valid_points:
        old_idx = (idx[0], idx[1], idx[2])

        for counter in range(6):
            new_idx = (idx[0], idx[1], idx[2])
            if counter == 0:
                # x+1
                new_idx = (idx[0] + 1, idx[1], idx[2])
            elif counter == 1:
                # x-1
                new_idx = (idx[0] - 1, idx[1], idx[2])
            elif counter == 2:
                # y+1
                new_idx = (idx[0], idx[1] + 1, idx[2])
            elif counter == 3:
                # y-1
                new_idx = (idx[0], idx[1] - 1, idx[2])
            elif counter == 4:
                # z+1
                new_idx = (idx[0], idx[1], idx[2] + 1)
            elif counter == 5:
                # z-1
                new_idx = (idx[0], idx[1], idx[2] - 1)
            else:
                raise Exception("Invalid counter")

            if new_idx not in valid_points:
                continue
            idx_i = points[old_idx]
            idx_j = points[new_idx]
            if idx_i != idx_j:
                graph[(idx_i, idx_j)] = 1
                graph[(idx_j, idx_i)] = 1

    return points, graph


def chunk_array(data, chuck_max_size):
    """
    Split array into chunks
    """
    return


def filter_interior_points(data):
    """
    Masks locations where the voxel has 8 neighbors

    :returns: surface_data
    """
    neighbors = get_neighbors(data)
    surface_data = np.zeros(data.shape)
    with_lt_8_neighbors = np.argwhere(neighbors < 8)
    for idx in with_lt_8_neighbors:
        if data[(idx[0], idx[1], idx[2])] == 1:
            surface_data[(idx[0], idx[1], idx[2])] = 1

    return surface_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='computes specific area')
    parser.add_argument('--working_dir', help='bmp files directory', required=True)

    args = parser.parse_args()
    working_dir = args.working_dir
    im_files = sorted([os.path.join(working_dir, f) for f in os.listdir(working_dir) if f.endswith(".bmp")])
    n_files = len(im_files)
    data = geometry.load_images_to_logical_array(im_files, x_lims=(0, 5),y_lims=(0, 5), z_lims=(0, 5))
    surface_data = filter_interior_points(data)
    points, B = build_graph(surface_data)
    L = np.matmul(B, B.transpose())
    ns1 = np.around(linalg.null_space(B.transpose()), 0)
    print(ns1)
    ns = linalg.null_space(L)
    print("Number of pieces:", ns.shape[1])