#!/usr/bin/env python3

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from skimage import io

import constants, geometry


def load_images(files_list, shape):
    """"""
    Nx, Ny, Nz = shape
    data = np.zeros([int(Nx), int(Ny), int(Nz)], dtype=np.uint8)
    for i, img_file in enumerate(sorted(files_list)):
        img = plt.imread(img_file)
        data[:, :, i] = img

    return data


def piece_is_collinear(piece_coordinates):
    """"""
    return


def piece_is_planar(piece_coordinates):
    """
    """
    x_values = set()
    y_values = set()
    z_values = set()

    for coord in piece_coordinates:
        x, y, z = coord
        x_values.add(x)
        y_values.add(y)
        z_values.add(z)
    
    return np.array([len(x_values) == 1, len(y_values) == 1, len(z_values) == 1]).any()


def piece_is_wholly_surrounded_by_one_different_phase(piece_coordinates, piece_phase, voxels):
    """"""
    surrounding_phases = set()
    for coord in piece_coordinates:
        x, y, z = coord
        surrounding_phases.add(voxels[coord])
        neighbors = [
            (int(x + 1), y, z),
            (int(x - 1), y, z),
            (x, int(y + 1), z),
            (x, int(y - 1), z),
            (x, y, int(z + 1)),
            (x, y, int(z - 1)),
        ]
        for p in neighbors:
            try:
                surrounding_phases.add(
                    voxels[p]
                )
            except IndexError:
                continue
    surrounding_phases.remove(piece_phase)

    return len(surrounding_phases) == 1, surrounding_phases.pop()


def get_filtered_voxels(voxels):
    """"""
    filtered = voxels
    for phase in constants.PHASES:
        phase_voxels = voxels == phase
        points = geometry.build_points(phase_voxels)
        points_view = {v: k for k, v in points.items()}
        G = geometry.build_graph(points)
        pieces = nx.connected_components(G)
        pieces = [piece for piece in pieces]
        for piece in pieces:
            piece_coordinates = [points_view[p] for p in piece]
            if piece_is_planar(piece_coordinates):
                submerged, submerging_phase = piece_is_wholly_surrounded_by_one_different_phase(piece_coordinates, phase, voxels)
                if submerged:
                    np.put(filtered, piece_coordinates, submerging_phase)

    return filtered