#!/usr/bin/env python3

import os
import timeit

import argparse
import dolfinx
import logging
import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc

import commons, configs, constants


markers = commons.SurfaceMarkers()

# Some constants
D_am = 5e-15
D_se = 0
# electronic conductivity
sigma_am = 5e3
sigma_se = 0
# ionic conductivity
kappa_am = 0
kappa_se = 0.1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument('--grid_size', help='Lx-Ly-Lz', required=True)
    parser.add_argument('--data_dir', help='Directory with tria.xdmf and tetr.xmf mesh files. Output files potential.xdmf and current.xdmf will be saved here', required=True, type=str)
    parser.add_argument("--voltage", help="Potential to set at the left current collector. Right current collector is set to a potential of 0", nargs='?', const=1, default=1)

    args = parser.parse_args()
    data_dir = args.data_dir
    voltage = args.voltage
    comm = MPI.COMM_WORLD
    rank = comm.rank
    start_time = timeit.default_timer()
    scaling = configs.get_configs()['VOXEL_SCALING']
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])
    loglevel = configs.get_configs()['LOGGING']['level']

    grid_size = args.grid_size
    FORMAT = f'%(asctime)s: %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(f'{data_dir}')
    logger.setLevel(loglevel)
    Lx, Ly, Lz = [int(v) for v in grid_size.split("-")]
    Lx = Lx * scale_x
    Ly = Ly * scale_y
    Lz = Lz * scale_z
    tetr_mesh_path = os.path.join(data_dir, 'tetr.xdmf')
    tria_mesh_path = os.path.join(data_dir, 'tria.xdmf')
    output_current_path = os.path.join(data_dir, 'current.xdmf')
    output_potential_path = os.path.join(data_dir, 'potential.xdmf')

    left_cc_marker = markers.left_cc
    right_cc_marker = markers.right_cc
    insulated_marker = markers.insulated

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh/laminate/tria.xdmf", "r") as xdmf:
        mesh2d = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(mesh2d, name="Grid")

    mesh2d.topology.create_connectivity(dolfinx.mesh.topology.dim, dolfinx.mesh.topology.dim - 1)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh/laminate/line.xdmf", "r") as xdmf:
        mesh1d = xdmf.read_mesh(name="Grid")
        ft = xdmf.read_meshtags(mesh1d, name="Grid")

    Q = dolfinx.FunctionSpace(mesh2d, ("DG", 0))
    kappa = dolfinx.Function(Q)
    sigma = dolfinx.Function(Q)
    d_eff = dolfinx.Function(Q)
    se_cells = ct.find(1)
    am_cells = ct.find(3)
    kappa.x.array[am_cells] = np.full_like(am_cells, kappa_am, dtype=PETSc.ScalarType)
    kappa.x.array[se_cells]  = np.full_like(se_cells, kappa_se, dtype=PETSc.ScalarType)
    sigma.x.array[am_cells] = np.full_like(am_cells, sigma_am, dtype=PETSc.ScalarType)
    sigma.x.array[se_cells]  = np.full_like(se_cells, sigma_se, dtype=PETSc.ScalarType)
    d_eff.x.array[am_cells] = np.full_like(am_cells, D_am, dtype=PETSc.ScalarType)
    d_eff.x.array[se_cells]  = np.full_like(se_cells, D_se, dtype=PETSc.ScalarType)

    x = ufl.SpatialCoordinate(mesh2d)