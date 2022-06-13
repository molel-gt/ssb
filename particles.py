#! /usr/bin/env python3

import csv
import gc
import os
import subprocess

import argparse
# import matlab.engine
import matplotlib.pyplot as plt
import meshio
import networkx as nx
import numpy as np

from collections import defaultdict
from PIL import Image
from mpi4py import MPI
from scipy import linalg
from scipy.io import loadmat, savemat

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
    2. ignore pieces with points all on the same plane,
       e.g. {(x1, y1, z0), (x1, y2, z0), (x3, y1, z0), (x2, y1, z0)}

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


def build_piece_matrix(data, idx, fname):
    """"""
    piece = {f"p{idx}": data}
    savemat(fname, piece)
    return


def save_solid_piece_to_file(piece, points_view, shape, idx, fname):
    """"""
    data = np.zeros(shape, dtype=int)
    for point in piece:
        coord = points_view[point]
        data[coord] = 1
    build_piece_matrix(data, idx, fname)
    return


def particle_neighborhood(data, particle_surf_points, points_view, particle_phase):
    """
    :param data:
    :param particle_surf_points:
    :param particle_phase:
    :param other_phase:
    """
    neighbor_points = set()
    neighborhood = defaultdict(lambda: 0)
    for point_idx in particle_surf_points:
        x, y, z = points_view[point_idx]
        neighbors = [(int(x - 1), int(y), int(z)), (int(x + 1), int(y), int(z)),
                     (int(x), int(y - 1), int(z)), (int(x), int(y + 1), int(z)),
                     (int(x), int(y), int(z - 1)), (int(x), int(y), int(z + 1))
                    ]
        for neighbor in neighbors:
            # try block for out of index access
            try:
                value = data[neighbor]
                if neighbor not in neighbor_points and value != particle_phase:
                    neighborhood[value] += 1
                    neighbor_points.add(neighbor)
            except IndexError:
                continue
    return neighborhood


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='computes specific area')
#     parser.add_argument('--img_folder', help='bmp files directory',
#                         required=True)
#     parser.add_argument('--grid_info', help='Nx-Ny-Nz',
#                         required=True)
#     parser.add_argument('--origin', default=(0, 0, 0), help='where to extract grid from')

#     args = parser.parse_args()
#     if isinstance(args.origin, str):
#         origin = tuple(map(lambda v: int(v), args.origin.split(",")))
#     else:
#         origin = args.origin
#     origin_str = "_".join([str(v) for v in origin])
#     grid_info = args.grid_info
#     grid_size = int(args.grid_info.split("-")[0])
#     Nx, Ny, Nz = [int(v) for v in args.grid_info.split("-")]
#     img_dir = args.img_folder
#     im_files = sorted([os.path.join(img_dir, f) for
#                        f in os.listdir(img_dir) if f.endswith(".bmp")])
#     n_files = len(im_files)

#     data = geometry.load_images_to_voxel(im_files, x_lims=(0, Nx),
#                                          y_lims=(0, Ny), z_lims=(0, Nz), origin=origin)

#     surface_data = filter_interior_points(data)
#     # pad data with extra row and column to allow +1 out-of-index access
#     data_padded = np.zeros((Nx + 1, Ny + 1, Nz + 1))
#     data_padded[0:Nx, 0:Ny, 0:Nz] = surface_data
#     points, G = build_graph(data_padded)
#     points_view = {v: k for k, v in points.items()}

#     print("Getting connected pieces..")
#     solid_pieces = [p for p in get_connected_pieces(G) if is_piece_solid(p, points_view)]

#     # Summary
#     print("Grid: {}x{}x{}".format(*[int(v) for v in data.shape]))
#     print("Number of pieces:", len(solid_pieces))
#     tif_files = sorted([os.path.join(img_dir, '../', f) for
#                        f in os.listdir(os.path.join(img_dir, '../')) if f.endswith(".tif")])
#     tif_data = np.zeros((len(tif_files), 451, 801), dtype=np.uint8)
#     for idx, tif_file in enumerate(tif_files):
#         tif_data[idx, :, :] = np.array(Image.open(tif_file), dtype=np.uint8)
#     matlab_eng = matlab.engine.start_matlab()
#     voxels = matlab_eng.GetVoxels('activematerial')
#     volume = matlab_eng.sum(voxels, 'all')
#     am_surface_area = matlab_eng.SurfArea(voxels)
#     print("Nominal active material surface area:", int(am_surface_area))
#     print("Active material volume:", int(volume))
#     print("Nominal active material specific area:", np.around(am_surface_area / volume, 4))
#     areas = {}
#     for idx, piece in enumerate(solid_pieces):
#         neighborhood = particle_neighborhood(tif_data, piece, points_view, np.uint8(1))
#         neighborhood_ratios = {}
#         total = 0
#         for k, v in neighborhood.items():
#             total += v
#         for k, v in neighborhood.items():
#             neighborhood_ratios[k] = v / total
#         void_in_one_phase = False
#         for k, v in neighborhood_ratios.items():
#             if np.isclose(v, 1):
#                 void_in_one_phase = True
#         # do not calculate area of voids that are wholly in electrolyte or active material
#         if void_in_one_phase:
#             continue
#         piece_data = np.zeros(data.shape, dtype=np.uint8)
#         for point_idx in piece:
#             piece_data[points_view[point_idx]] = 1
#         fname = os.path.join(args.img_folder, f"p{idx}.mat")
#         build_piece_matrix(piece_data, idx, fname)
#         var_name = f'p{idx}'
#         mat = matlab_eng.load(fname)
#         area = matlab_eng.SurfArea(mat[var_name])
#         areas[idx] = {"area": area, "ratio": neighborhood_ratios}
#         os.remove(fname)
#     am_void_area = 0
#     for k, v in areas.items():
#         am_void_area += v["area"] * v["ratio"][0]

#     print("AM area fraction covered by voids:", np.around(am_void_area / am_surface_area, 4))
#     with open("void-areas.csv", "w") as fp:
#         writer = csv.DictWriter(fp, fieldnames=["piece_idx", "area", "am_ratio"])
#         writer.writeheader()
#         for k, v in areas.items():
#             writer.writerow({"piece_idx": k, "area": np.around(v["area"], 0), "am_ratio": np.around(v["ratio"][0], 4)})
