#!/usr/bin/env python3

import os

import argparse
import matplotlib.pyplot as plt
import meshio
import numpy as np
import subprocess

import particles
import geometry


if __name__ == '__main__':
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

    data = geometry.load_images_to_logical_array(im_files, x_lims=(0, grid_size),
                                                 y_lims=(0, grid_size), z_lims=(0, grid_size))
    data = np.logical_not(data)
    Nx, Ny, Nz = data.shape
    surface_data = particles.filter_interior_points(data)
    # pad data with extra row and column to allow +1 out-of-index access
    data_padded = np.zeros((Nx + 1, Ny + 1, Nz + 1))
    data_padded[0:Nx, 0:Ny, 0:Nz] = data
    points, G = particles.build_graph(data_padded)
    points_view = {v: k for k, v in points.items()}

    pieces = particles.get_connected_pieces(G)
    solid_pieces = [p for p in pieces if particles.is_piece_solid(p, points_view)]

    for idx, piece in enumerate(solid_pieces):
        mshfile = particles.meshfile(piece, points_view, data.shape,
                (f"mesh/p{idx}.node", f"mesh/p{idx}.geo", f"mesh/p{idx}.vtk", f"mesh/p{idx}.msh"))

        # build .xdmf/.h5 file from .msh file
        msh = meshio.read(mshfile)
        print("creating tetrahedral mesh")  
        tetra_mesh = geometry.create_mesh(msh, "tetra")
        meshio.write(f"mesh/p{idx}_tetr.xdmf", tetra_mesh)
