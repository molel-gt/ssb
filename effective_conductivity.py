#!/usr/bin/env python3

import os

import argparse
import h5py
import numpy as np

import geometry


def kappa_eff(i_cell, eps_sse, delta_phi):
    """
    Effective conductivity
    """
    return eps_sse * i_cell / (-delta_phi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run simulation..')
    parser.add_argument('--img_folder', help='bmp files sub directory', required=True)
    parser.add_argument('--grid_info', help='Nx-Ny-Nz', required=True)

    args = parser.parse_args()
    grid_info = args.grid_info
    current_file_path = os.path.join("output", grid_info + "_current.h5")
    current_data = h5py.File(current_file_path, "r")
    geom = current_data["/Mesh/Grid/geometry"]
    values = current_data["/Function/f/0"]
    Nx, Ny, Nz = [int(v) for v in grid_info.split("-")]
    data = np.zeros((Nx, Ny, Nz), dtype=np.double)
    for idx, value in enumerate(values):
        coord = [int(v) for v in geom[idx]]
        data[tuple(coord)] = value
    files_list = sorted([os.path.join(args.img_folder, f) for f in os.listdir(args.img_folder)
                  if f.endswith(".bmp")])
    image_data = geometry.load_images_to_voxel(files_list, (0, int(Nx)), (0, int(Ny)), (0, int(Nz)))
    porosity = np.average(image_data)
    delta_phi = -1 / (Nx - 1)
    current0 = np.average(data[0, :, :])
    current_end = np.average(data[int(Nx - 1), :, :])
    i_cell = 0.5 * (current0 + current_end)
    # results summary
    print("Porosity:", np.around(porosity, 4))
    print("Bruggeman:", np.around(porosity ** 1.5, 4))
    print("Model:", np.around(kappa_eff(i_cell, porosity, delta_phi), 4))
