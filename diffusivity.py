#!/usr/bin/env python3

import os
import time

import argparse
import logging
import numpy as np
import ufl

from dolfinx import cpp, fem, io, mesh, nls, plot
from mpi4py import MPI
from petsc4py import PETSc

import commons, configs, geometry, utils

markers = commons.SurfaceMarkers()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Diffusivity')
    parser.add_argument('--grid_extents', help='Nx-Ny-Nz_Ox-Oy-Oz size_location', required=True)
    parser.add_argument('--root_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage", nargs='?', const=1, default=1e-3)
    parser.add_argument("--scale", help="sx,sy,sz", nargs='?', const=1, default='-1,-1,-1')
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='VOXEL_SCALING', type=str)
    args = parser.parse_args()
    loglevel = configs.get_configs()['LOGGING']['level']
    grid_extents = args.grid_extents
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    formatter = logging.Formatter(f'%(levelname)s:%(asctime)s:{grid_extents}:%(message)s')
    fh = logging.FileHandler('transport.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
