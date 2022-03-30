#! /usr/bin/env python3

import os
import subprocess

import argparse
import matplotlib.pyplot as plt
import meshio
import networkx as nx
import numpy as np

from collections import defaultdict
from mpi4py import MPI
from scipy import linalg
from scipy.io import savemat

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


def center_of_mass(piece, points_view):
    x_cm = y_cm = z_cm = 0
    n = len(piece)
    for point in piece:
        x, y, z = points_view[point]
        x_cm += x / n
        y_cm += y / n
        z_cm += z / n

    return x_cm, y_cm, z_cm


def sphericity(V_p, A_p):
    """
    param V_p: particle volume
    param A_p: particle surface area
    """
    return ((np.pi) ** (1/3) ) * ((6 * V_p) ** (2/3)) / A_p


def meshfile(piece, points_view, shape, file_names):
    """
    file_names =: (node, geo, vtk, msh) files
    """
    data = np.zeros(shape)
    for idx in piece:
        coord = points_view[idx]
        data[coord] = True
    nodes = geometry.create_nodes(data)
    geometry.write_node_to_file(nodes, file_names[0])
    _ = subprocess.check_call("./nodes_to_msh.sh %s %s %s %s" % file_names, shell=True)

    return file_names[-1]


def build_piece_matrix(data, piece_idx):
    """"""
    piece = {"data": data, "label": "piece"}
    savemat(f"{piece_idx}.mat", piece)
    return


def save_solid_piece_to_file(piece, points_view, shape, fname):
    """"""
    data = np.zeros(shape, dtype=int)
    for point in piece:
        coord = points_view[point]
        data[coord] = 1
    build_piece_matrix(data, fname.strip(".dat"))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='computes specific area')
    parser.add_argument('--img_dir', help='bmp files directory',
                        required=True)
    parser.add_argument('--grid_info', help='Nx-Ny-Nz',
                        required=True)

    args = parser.parse_args()
    grid_size = int(args.grid_info.split("-")[0])
    img_dir = args.img_dir
    im_files = sorted([os.path.join(img_dir, f) for
                       f in os.listdir(img_dir) if f.endswith(".bmp")])
    n_files = len(im_files)

    data = geometry.load_images_to_voxel(im_files, x_lims=(0, grid_size),
                                                 y_lims=(0, grid_size), z_lims=(0, grid_size))
    data = np.logical_not(data)
    Nx, Ny, Nz = data.shape
    surface_data = filter_interior_points(data)
    # pad data with extra row and column to allow +1 out-of-index access
    data_padded = np.zeros((Nx + 1, Ny + 1, Nz + 1))
    data_padded[0:Nx, 0:Ny, 0:Nz] = surface_data
    points, G = build_graph(data_padded)
    points_view = {v: k for k, v in points.items()}

    pieces = get_connected_pieces(G)
    solid_pieces = [p for p in pieces if is_piece_solid(p, points_view)]
    for idx, piece in enumerate(solid_pieces):
        save_solid_piece_to_file(piece, points_view, data.shape, os.path.join('spheres', str(idx).zfill(3) + '.dat'))
    centers_of_mass = [center_of_mass(p, points_view) for p in solid_pieces]

    # Summary
    print("Grid: {}x{}x{}".format(*[int(v) for v in data.shape]))
    print("Number of pieces:", len(solid_pieces))
    print("Centers of mass:", np.around(centers_of_mass, 2))
