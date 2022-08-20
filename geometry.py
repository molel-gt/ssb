#!/usr/bin/env python3

import os

import argparse
import meshio
import numpy as np
import subprocess

from skimage import io

import utils


OFFSET = 2


def load_images_to_voxel(files_list, x_lims=(0, 201), y_lims=(0, 201), z_lims=(0, 201), origin=(0, 0, 0), phase=0):
    """
    grid_sizes: Lx.Ly.Lz
    """
    x0, x1 = x_lims
    y0, y1 = y_lims
    z0, z1 = z_lims
    Lx = x1 - x0
    Ly = y1 - y0
    Lz = z1 - z0
    dx, dy, dz = origin
    data = np.zeros([int(Lx), int(Ly), int(Lz)], dtype=bool)
    for i_x, img_file in enumerate(files_list):
        if not (x0 + dx <= i_x <= x1 + dx):
            continue
        img_data = io.imread(img_file)
        img_data = img_data == (phase + OFFSET)
        data[i_x - x0 - dx - 1, :, :] = img_data[int(dy + y0):int(y1 + dy), int(dz + z0):int(dz + z1)]
    return data


def create_mesh(mesh, cell_type, prune_z=False):
    """
    """
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points,
                           cells={cell_type: cells},
                           cell_data={"name_to_read": [cell_data]}
                           )
    if prune_z:
        out_mesh.prune_z_0()
    return out_mesh
