#! /usr/bin/env python3

import os

import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from collections import defaultdict
from scipy import linalg

import geometry

AREA_WEIGHTS = {
    0: 0,
    1: 0.636,
    2: 0.669,
    3: 1.272,
    4: 1.272,
    5: 0.554,
    6: 1.305,
    7: 1.908,
    8: 0.927,
    9: 0.442,
    10: 1.338,
    11: 1.573,
    12: 1.190,
    13: 2.544,
    14: 1.573,
}
CASES = {
    (False, False, False, False, False, False, False, False): 0,
    # case 1 config
    (True, False, False, False, False, False, False, False): 1,
    (False, True, False, False, False, False, False, False): 1,
    (False, False, True, False, False, False, False, False): 1,
    (False, False, False, True, False, False, False, False): 1,
    # case 2 config
    (True, True, False, False, False, False, False, False): 2,
    (False, True, True, False, False, False, False, False): 2,
    (False, False, True, True, False, False, False, False): 2,
    (True, False, False, True, False, False, False, False): 2,
    ##
    (True, False, True, False, False, False, False, False): 3,
    (True, False, False, False, True, False, False, False): 3,
    (True, False, False, False, False, False, True, False): 3,

    (False, True, False, True, False, False, False, False): 3,
    (False, True, False, False, False, True, False, False): 3,
    (False, True, False, False, False, False, False, True): 3,

    (False, False, True, False, True, False, False, False): 3,
    (False, False, True, False, False, False, True, False): 3,

    (False, False, False, True, False, True, False, False): 3,
    (False, False, False, True, False, False, False, True): 3,

    (False, False, False, True, False, False, True, False): 4,
    ##
    (True, True, True, False, False, False, False, False): 5,
    (False, True, True, True, False, False, False, False): 5,
    (True, False, True, True, False, False, False, False): 5,
    (True, True, False, True, False, False, False, False): 5,

    (False, False, False, False, False, True, True, True): 5,
    (False, False, False, False, False, True, True, True): 5,
    (False, False, False, False, True, False, True, True): 5,
    (False, False, False, False, True, True, False, True): 5,
    ##
    (False, True, True, False, False, False, False, True): 6,
    (True, False, True, False, False, False, True, False): 7,
    ##
    (True, True, True, True, False, False, False, False): 8,
    (False, False, False, False, True, True, True, True): 8,
    (True, True, False, False, False, True, True, False): 8,
    (False, False, True, True, True, False, False, True): 8,
    (False, True, True, False, False, False, True, True): 8,
    (True, False, False, True, True, True, False, False): 8,
    ##
    (True, True, True, False, False, False, True, False): 9,
    (False, True, True, True, False, False, False, True): 9,
    (True, False, True, True, True, False, False, False): 9,
    (True, True, False, True, False, True, False, False): 9,
    ##
    (False, True, True, False, True, False, False, True): 10,
    (True, True, True, False, False, False, False, True): 11,
    (True, True, True, False, True, False, False, False): 12,
    (False, True, False, True, False, True, False, True): 13,
    (True, True, False, True, False, False, True, False): 14,
}

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
    G = nx.Graph()
    points = defaultdict(lambda: -1, {})
    valid_points = set([tuple(v) for v in np.argwhere(array_chunk == 1)])
    for idx, value in enumerate(valid_points):
        points[(value[0], value[1], value[2])] = idx
        G.add_node(idx)

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
                G.add_edge(idx_i, idx_j)

    return points, G


def chunk_array(data, chuck_max_size):
    """
    Split array into chunks
    """
    return


def build_2x2x2_cube(idx):
    """
    Build 2x2x2 cube for marching cubes algorithm
    :return: cubepoints
    :rtype: list
    """
    x0, y0, z0 = idx
    cubepoints = np.zeros((8, 3))
    cubepoints[0, :] = (x0, y0, z0)
    cubepoints[1, :] = (int(x0 + 1), y0, z0)
    cubepoints[2, :] = (int(x0 + 1), int(y0 + 1), z0)
    cubepoints[3, :] = (x0, int(y0 + 1), z0)
    cubepoints[4, :] = (x0, int(y0 + 1), int(z0 + 1))
    cubepoints[5, :] = (x0, y0, int(z0 + 1))
    cubepoints[6, :] = (int(x0 + 1), y0, int(z0 + 1))
    cubepoints[7, :] = (int(x0 + 1), int(y0 + 1), int(z0 + 1))

    return cubepoints


def categorize_area_cases(cubepoints, data):
    """
    Categorize the points according to cases reported by Lindblad 2005.

    :return: case
    :rtype: int
    """
    search_key = tuple([data[(int(p[0]), int(p[1]), int(p[2]))] == 1 for p in cubepoints])
    case = CASES.get(search_key)
    if search_key is None:
        print(search_key)
    
    return case


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


def get_connected_pieces(G):
    """"""
    pieces = []
    for item in nx.connected_components(G):
        pieces.append(item)
    
    return pieces


def is_piece_solid(S, points_view):
    """
    Rules for checking if piece encloses a solid:
    1. ignore pieces with <= 3 points as they cannot enclose a solid
    2. ignore pieces with points all on the same plane, e.g. {(x1, y1, z0), (x1, y2, z0), (x3, y1, z0), (x2, y1, z0)}

    :rtype: bool
    """
    if len(S) <= 3:
        return False
    # check if values are on same plane
    x_values = set()
    y_values = set()
    z_values = set()
    for val in S:
        x, y, z = points_view[val]
        x_values.add(x)
        y_values.add(y)
        z_values.add(z)

    if len(x_values) <= 1 or len(y_values) <= 1 or len(z_values) <= 1:
        return False
    # TODO: Add further checks of connectivity to enclose a solid
    return True


def surface_area(cluster, data, points_view):
    """"""
    num_cases = {k: 0 for k in range(15)}
    for point in cluster:
        cubepoints = build_2x2x2_cube(points_view[point])
        case = categorize_area_cases(cubepoints, data)
        if case is None:
            continue
        num_cases[case] += 1
    surface_area = 0
    for k, num in num_cases.items():
        surface_area += AREA_WEIGHTS[k] * num

    return surface_area


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='computes specific area')
    parser.add_argument('--working_dir', help='bmp files directory',
                        required=True)
    parser.add_argument('--size', help='grid size',
                        required=True)

    args = parser.parse_args()
    grid_size = int(args.size)
    working_dir = args.working_dir
    im_files = sorted([os.path.join(working_dir, f) for
                       f in os.listdir(working_dir) if f.endswith(".bmp")])
    n_files = len(im_files)
    # solid electrolyte: true
    # active material/void: false
    data = geometry.load_images_to_logical_array(im_files, x_lims=(0, grid_size),
                                                 y_lims=(0, grid_size), z_lims=(0, grid_size))
    Nx, Ny, Nz = data.shape
    # data = np.logical_not(data)  # invert to focus on active material
    surface_data = filter_interior_points(data)
    # pad data with extra row and column to allow +1 out-of-index access
    data_padded = np.zeros((Nx + 1, Ny + 1, Nz + 1))
    data_padded[0:Nx, 0:Ny, 0:Nz] = surface_data
    points, G = build_graph(data_padded)
    points_view = {v: k for k, v in points.items()}

    # B = nx.adjacency_matrix(G).toarray()
    # L = nx.laplacian_matrix(G).toarray()
    # L_calc = np.matmul(B, B.transpose())
    # ns = linalg.null_space(L)

    pieces = get_connected_pieces(G)
    solid_pieces = [p for p in pieces if is_piece_solid(p, points_view)]
    areas = [np.around(surface_area(p, data_padded, points_view), 3) for p in solid_pieces]
    print("Grid: {}x{}x{}".format(*[int(v + 1) for v in data.shape]))
    print("Number of pieces:", len(solid_pieces))
    print("Areas:", sorted(areas, reverse=True))
    plt.plot(areas[1:], 'b-')
    plt.xlabel("rank")
    plt.ylabel("area")
    plt.title("Grid: {}x{}x{}".format(Nx + 1, Ny + 1, Nz + 1))
    plt.show()
