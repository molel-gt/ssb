#!/usr/bin/env python3
# coding: utf-8

import os

import argparse
import h5py
import logging
import numpy as np

import geometry, particles


FORMAT = '%(asctime)s: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel('INFO')


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

    voxels = geometry.load_images_to_voxel(im_files, x_lims=(0, Nx),
                                         y_lims=(0, Ny), z_lims=(0, Nz), origin=origin)
    eps_left = np.around(np.average(voxels[:, 0, :]), 4)
    eps_right = np.around(np.average(voxels[:, 50, :]), 4)
    eps = np.around(np.average(voxels), 4)
    logger.info("Porosity (y = 0)      : %s" % eps_left)
    logger.info("Porosity (y = Ly)     : %s" % eps_right)
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
    current_left = np.around(np.average(vals_left), 4)
    current_right = np.around(np.average(vals_right), 4)
    kappa_eff_left = np.around(eps_left * current_left * Ly, 4)
    kappa_eff_right = np.around(eps_right * current_right * Ly, 4)
    logger.info("kappa_eff (y = 0)     : %s" % kappa_eff_left)
    logger.info("kappa_eff (y = Ly)    : %s" % kappa_eff_right)
    logger.info("Porosity (avg)        : %s" % eps)
    logger.info("kappa_eff (bruggeman) : %s" % np.around(eps ** 1.5, 4))
    print(current_left, current_right)