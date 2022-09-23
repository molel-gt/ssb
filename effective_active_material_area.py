#!/usr/bin/env python3

import os
import timeit

import argparse
from typing import Mapping
import dolfinx
import logging
import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc

import constants, utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run simulation..')
    parser.add_argument('--grid_info', help='Nx-Ny-Nz', required=True)
    parser.add_argument('--data_dir', help='directory with mesh files. output will be saved here', required=True, type=str)
    parser.add_argument("--voltage", nargs='?', const=1, default=1)
    parser.add_argument("--scale_x", nargs='?', const=1, default=1, type=np.double)
    parser.add_argument("--scale_y", nargs='?', const=1, default=1, type=np.double)
    parser.add_argument("--scale_z", nargs='?', const=1, default=1, type=np.double)

    args = parser.parse_args()
    data_dir = args.data_dir
    voltage = args.voltage
    scale_x = args.scale_x
    scale_y = args.scale_y
    scale_z = args.scale_z
    comm = MPI.COMM_WORLD
    rank = comm.rank
    start_time = timeit.default_timer()

    grid_info = "-".join([v.zfill(3) for v in args.grid_info.split("-")])
    FORMAT = f'%(asctime)s: %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(f'{grid_info}')
    logger.setLevel('DEBUG')
    Nx, Ny, Nz = [int(v) for v in grid_info.split("-")]
    Lx = (Nx - 1) * scale_x
    Ly = (Ny - 1) * scale_y
    Lz = (Nz - 1) * scale_z
    tetr_mesh_path = os.path.join(data_dir, 'tetr.xdmf')
    tria_mesh_path = os.path.join(data_dir, 'tria.xdmf')

    logger.debug("Loading triangles (dim = 2) mesh..")
    with dolfinx.io.XDMFFile(comm, tria_mesh_path, "r") as infile3:
        mesh = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
        facets_ct = infile3.read_meshtags(mesh, name="Grid")

    active_marker = constants.surface_tags["active_area"]
    inactive_marker = constants.surface_tags["inactive_area"]
    active_facet = np.array(facets_ct.indices[facets_ct.values == active_marker])
    inactive_facet = np.array(facets_ct.indices[facets_ct.values == inactive_marker])

    surf_meshtags = dolfinx.mesh.meshtags(mesh, 2, facets_ct.indices, facets_ct.values)
    n = -ufl.FacetNormal(mesh)
    ds_active = ufl.Measure('dx', domain=mesh, subdomain_data=surf_meshtags, subdomain_id=active_marker)
    ds_inactive = ufl.Measure('dx', domain=mesh, subdomain_data=surf_meshtags, subdomain_id=inactive_marker)

    # Define variational problem
    x = ufl.SpatialCoordinate(mesh)

    logger.debug('Solving problem..')

    active_area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_active))
    inactive_area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_inactive))
    logger.info("**************************RESULTS-SUMMARY******************************************")
    logger.info("Effective Active Material Area [sq. um]                  : {:.4e}".format(active_area))
    logger.info("Ineffective Active Material Area [sq. um]                : {:.4e}".format(inactive_area))
    logger.info("Total Area [sq. um]                                      : {:.4e}".format(active_area + inactive_area))
    logger.info("Total Volume [cu. um]                                    : {:.4e}".format(Lx * Ly * Lz))
    logger.info("Specific Effective Active Material Area [sq. um]         : {:.4e}".format(active_area/(Lx * Ly * Lz)))
    logger.info("Specific Ineffective Active Material Area [sq. um]       : {:.4e}".format(inactive_area/(Lx * Ly * Lz)))
    logger.info(f"Time elapsed                                            : {int(timeit.default_timer() - start_time):3.5f}s")
    logger.info("*************************END-OF-SUMMARY*******************************************")
