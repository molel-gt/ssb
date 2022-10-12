#!/usr/bin/env python3

import os
import timeit

import argparse
import dolfinx
import logging
import numpy as np
import ufl

from mpi4py import MPI

import constants


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Specific Active Material Area.')
    parser.add_argument('--grid_size', help='Lx-Ly-Lz', required=True)
    parser.add_argument('--data_dir', help='Directory with tria.xdmf and tetr.xdmf mesh files, and effective_electrolyte.pickle file. Output files potential.xdmf and current.xdmf will be saved here.', required=True, type=str)
    parser.add_argument("--scale_x", help="Value to scale the Lx grid size given to match dimensions of mesh files.", nargs='?', const=1, default=1, type=np.double)
    parser.add_argument("--scale_y", help="Value to scale the Ly grid size given to match dimensions of mesh files.", nargs='?', const=1, default=1, type=np.double)
    parser.add_argument("--scale_z", help="Value to scale the Lz grid size given to match dimensions of mesh files.", nargs='?', const=1, default=1, type=np.double)

    args = parser.parse_args()
    data_dir = args.data_dir
    scale_x = args.scale_x
    scale_y = args.scale_y
    scale_z = args.scale_z
    comm = MPI.COMM_WORLD
    rank = comm.rank
    start_time = timeit.default_timer()

    grid_size = args.grid_size
    FORMAT = f'%(asctime)s: %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(f'{grid_size}' + '_' + __file__)
    logger.setLevel('DEBUG')
    Lx, Ly, Lz = [int(v) for v in grid_size.split("-")]
    tetr_mesh_path = os.path.join(data_dir, 'tetr.xdmf')
    tria_mesh_path = os.path.join(data_dir, 'tria.xdmf')

    logger.debug("Loading triangles (dim = 2) mesh..")
    with dolfinx.io.XDMFFile(comm, tria_mesh_path, "r") as infile3:
        mesh = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
        facets_ct = infile3.read_meshtags(mesh, name="Grid")

    left_cc_marker = constants.surface_tags["left_cc"]
    right_cc_marker = constants.surface_tags["right_cc"]
    active_marker = constants.surface_tags["active_area"]
    inactive_marker = constants.surface_tags["inactive_area"]

    # x0facet = np.array(facets_ct.indices[facets_ct.values == left_cc_marker])
    # x1facet = np.array(facets_ct.indices[facets_ct.values == right_cc_marker])
    # active_facet = np.array(facets_ct.indices[facets_ct.values == active_marker])
    # inactive_facet = np.array(facets_ct.indices[facets_ct.values == inactive_marker])

    surf_meshtags = dolfinx.mesh.meshtags(mesh, 2, facets_ct.indices, facets_ct.values)
    n = -ufl.FacetNormal(mesh)
    ds_left_cc = ufl.Measure('dx', domain=mesh, subdomain_data=surf_meshtags, subdomain_id=left_cc_marker)
    ds_right_cc = ufl.Measure('dx', domain=mesh, subdomain_data=surf_meshtags, subdomain_id=right_cc_marker)
    ds_active = ufl.Measure('dx', domain=mesh, subdomain_data=surf_meshtags, subdomain_id=active_marker)
    ds_inactive = ufl.Measure('dx', domain=mesh, subdomain_data=surf_meshtags, subdomain_id=inactive_marker)

    # Define variational problem
    x = ufl.SpatialCoordinate(mesh)

    logger.debug('Solving problem..')

    active_area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_active))
    inactive_area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_inactive))
    area_left_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_left_cc))
    area_right_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_right_cc))
    logger.info("**************************RESULTS-SUMMARY******************************************")
    logger.info("Contact Area @ left cc [sq. um]                          : {:.4e}".format(area_left_cc))
    logger.info("Contact Area @ right cc [sq. um]                         : {:.4e}".format(area_right_cc))
    logger.info("Effective Active Material Area [sq. um]                  : {:.4e}".format(active_area))
    logger.info("Ineffective Active Material Area [sq. um]                : {:.4e}".format(inactive_area))
    logger.info("Total Area [sq. um]                                      : {:.4e}".format(active_area + inactive_area))
    logger.info("Total Volume [cu. um]                                    : {:.4e}".format(Lx * Ly * Lz))
    logger.info("Specific Effective Active Material Area [sq. um]         : {:.4e}".format(active_area/(Lx * Ly * Lz)))
    logger.info("Specific Ineffective Active Material Area [sq. um]       : {:.4e}".format(inactive_area/(Lx * Ly * Lz)))
    logger.info(f"Time elapsed                                             : {int(timeit.default_timer() - start_time):3.5f}s")
    logger.info("*************************END-OF-SUMMARY*******************************************")