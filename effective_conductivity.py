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
    parser.add_argument('--origin', default=(0, 0, 0), help='where to extract grid from')

    args = parser.parse_args()
    if isinstance(args.origin, str):
        origin = tuple(map(lambda v: int(v), args.origin.split(",")))
    else:
        origin = args.origin
    origin_str = "_".join([str(v) for v in origin])
    grid_info = args.grid_info
    current_file_path = os.path.join("output", 's'+grid_info+'o'+origin_str + "_current.h5")
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
    image_data = geometry.load_images_to_voxel(files_list, (0, int(Nx)), (0, int(Ny)), (0, int(Nz)), origin)
    eps_sse = np.average(image_data)
    delta_phi = -1 / (Nx - 1)
    current_at_0 = np.sum(data[0, :, :]) / ((Ny - 1) * (Nz - 1))
    current_at_Lx = np.sum(data[int(Nx - 1), :, :]) / ((Ny - 1) * (Nz - 1))

    eps_sse_at_0 = np.average(image_data[0, :, :])
    eps_sse_at_Lx = np.average(image_data[int(Nx - 1), :, :])
    kappa_eff_vals = []
    for i in range(Nx):
        current_val = np.sum(data[i, :, :]) / ((Ny - 1) * (Nz - 1))
        eps_val = np.average(image_data[i, :, :])
        kappa_eff_vals.append(kappa_eff(current_val, eps_val, delta_phi))

    # results summary
    print("Porosity @ x = 0                          :", np.around(eps_sse_at_0, 4))
    print("Porosity @ x = Lx                         :", np.around(eps_sse_at_Lx, 4))
    print("Avg. Porosity                             :", np.around(eps_sse, 4))
    print("Model @x = 0                              :", np.around(kappa_eff(current_at_0, eps_sse_at_0, delta_phi), 4))
    print("Model @x = Lx                             :", np.around(kappa_eff(current_at_Lx, eps_sse_at_Lx, delta_phi), 4))
    print("Model avg. current and avg. porosity      :", np.around(kappa_eff(np.average(values), eps_sse, delta_phi), 4))
    print("Model weighted average                    :", np.around(np.average(kappa_eff_vals), 4))
    print("Bruggeman                                 :", np.around(eps_sse ** 1.5, 4))
