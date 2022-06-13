#!/usr/bin/env python3
# coding: utf-8

import os

import argparse
import h5py
import numpy as np

import geometry, particles


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='computes specific area')
    parser.add_argument('--img_folder', help='bmp files directory',
                        required=True)
    parser.add_argument('--grid_info', help='Nx-Ny-Nz',
                        required=True)
    parser.add_argument('--origin', default=(0, 0, 0), help='where to extract grid from')

    args = parser.parse_args()
    if isinstance(args.origin, str):
        origin = tuple(map(lambda v: int(v), args.origin.split(",")))
    else:
        origin = args.origin
    origin_str = "_".join([str(v) for v in origin])
    grid_info = args.grid_info
    grid_size = int(args.grid_info.split("-")[0])
    Nx, Ny, Nz = [int(v) for v in args.grid_info.split("-")]
    img_dir = args.img_folder
    im_files = sorted([os.path.join(img_dir, f) for
                       f in os.listdir(img_dir) if f.endswith(".bmp")])
    n_files = len(im_files)

    data = geometry.load_images_to_voxel(im_files, x_lims=(0, Nx),
                                         y_lims=(0, Ny), z_lims=(0, Nz), origin=origin)

    # surface_data = particles.filter_interior_points(data)
    # # pad data with extra row and column to allow +1 out-of-index access
    # data_padded = np.zeros((Nx + 1, Ny + 1, Nz + 1))
    # data_padded[0:Nx, 0:Ny, 0:Nz] = surface_data
    # points, G = particles.build_graph(data_padded)
    # points_view = {v: k for k, v in points.items()}

    # print("Getting connected pieces..")
    # solid_pieces = [p for p in particles.get_connected_pieces(G) if particles.is_piece_solid(p, points_view)]
    # largest_piece = solid_pieces[0]
    new_data = data # np.zeros((Nx, Ny, Nz), dtype=np.bool8)
    # for pk in largest_piece:
    #     p = points_view[pk]
    #     new_data[p] = 1
    eps_left = np.around(np.average(new_data[:, 0, :]), 5)
    eps_right = np.around(np.average(new_data[:, 50, :]), 5)
    eps = np.around(np.average(new_data), 5)
    print("Porosity (y = 0): ", eps_left)
    print("Porosity (y = Ly): ", eps_right)
    print("Porosity (avg): ", eps)
    fname = f'output/s{grid_info}o{origin_str}_current.h5'
    Lx = int(Nx - 1)
    Ly = int(Ny - 1)
    Lz = int(Nz - 1)
    resultsdata = h5py.File(fname, "r")
    values = resultsdata['/Function/f/0']
    geometry = resultsdata['/Mesh/Grid/geometry']
    vals_left = []
    vals_right = []
    for idx, (vx, vy, vz) in enumerate(values):
        coord = geometry[idx, :]
        if np.isclose(coord[1], 0):
            mag = (vx ** 2 + vy ** 2 + vz ** 2) ** (0.5)
            vals_left.append(mag)
        if np.isclose(coord[1], Ly):
            mag = (vx ** 2 + vy ** 2 + vz ** 2) ** (0.5)
            vals_right.append(mag)
    current_left = np.around(np.average(vals_left), 5)
    current_right = np.around(np.average(vals_right), 5)
    kappa_eff_left = np.around(eps_left * current_left * Ly, 5)
    kappa_eff_right = np.around(eps_right * current_right * Ly, 5)
    print("kappa_eff (y = 0): ", kappa_eff_left)
    print("kappa_eff (y = Ly): ", kappa_eff_right)
    print("kappa_eff (bruggeman): ", np.around(eps ** 1.5, 5))