#!/usr/bin/env python3

import os
import pickle
import timeit

import argparse
import dolfinx
import logging
import numpy as np
import ufl

from collections import namedtuple
from mpi4py import MPI

import commons, constants


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

    logger.debug("Loading tetrahedra (dim = 3) mesh..")

    with dolfinx.io.XDMFFile(comm, tetr_mesh_path, "r") as infile3:
        mesh3d = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
        ct = infile3.read_meshtags(mesh3d, name="Grid")

    mesh3d.topology.create_connectivity(mesh3d.topology.dim, mesh3d.topology.dim - 1)

    x = ufl.SpatialCoordinate(mesh3d)

    # logger.debug("Loading triangles (dim = 2) mesh..")
    # with dolfinx.io.XDMFFile(comm, tria_mesh_path, "r") as infile3:
    #     mesh2d = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
    #     facets_ct = infile3.read_meshtags(mesh2d, name="Grid")

    # x = ufl.SpatialCoordinate(mesh2d)

    logger.debug("Loading effective electrolyte..")
    with open(os.path.join(data_dir, 'effective_electrolyte.pickle'), 'rb') as handle:
        effective_electrolyte = list(pickle.load(handle))

    left_cc_marker = constants.surface_tags["left_cc"]
    right_cc_marker = constants.surface_tags["right_cc"]
    insulated_marker = constants.surface_tags["insulated"]
    active_marker = constants.surface_tags["active_area"]
    inactive_marker = constants.surface_tags["inactive_area"]

    x0facet = dolfinx.mesh.locate_entities_boundary(mesh3d, 2, lambda x: np.isclose(x[1], 0.0))
    x1facet = dolfinx.mesh.locate_entities_boundary(mesh3d, 2, lambda x: np.isclose(x[1], Ly))
    insulated_facet = dolfinx.mesh.locate_entities_boundary(mesh3d, 2, lambda x: np.logical_and(np.logical_not(np.isclose(x[1], 0)), np.logical_not(np.isclose(x[1], Ly))))

    def is_active_area(x, effective_electrolyte, dp=1):
        ret_val = np.zeros(x.shape[1])
        for idx in range(x.shape[1]):
            c = tuple(x[:, idx])
            coord = (round(c[0]/scale_x, dp), round(c[1]/scale_y, dp), round(c[2]/scale_z, dp))
            ret_val[idx] = coord in effective_electrolyte
        ret_val.reshape(-1, 1)
        return ret_val

    active_facet = dolfinx.mesh.locate_entities_boundary(mesh3d, 2, lambda x: is_active_area(x, effective_electrolyte))
    inactive_facet = dolfinx.mesh.locate_entities_boundary(mesh3d, 2, lambda x: np.logical_not(is_active_area(x, effective_electrolyte)))

    facets_ct_indices1 = np.hstack((x0facet, x1facet, insulated_facet))
    facets_ct_values1 = np.hstack((np.ones(x0facet.shape[0], dtype=np.int32), right_cc_marker * np.ones(x1facet.shape[0], dtype=np.int32),
                                insulated_marker * np.ones(insulated_facet.shape[0], dtype=np.int32)))
    facets_ct_indices2 = np.hstack((active_facet, inactive_facet))
    facets_ct_values2 = np.hstack((active_marker * np.ones(active_facet.shape[0], dtype=np.int32), inactive_marker * np.ones(inactive_facet.shape[0], dtype=np.int32)))
    facets_ct_indices = np.hstack((facets_ct_indices1, facets_ct_indices2))
    facets_ct_values = np.hstack((facets_ct_values1, facets_ct_values2))
    facets_ct = commons.Facet(facets_ct_indices, facets_ct_values)
    surf_meshtags = dolfinx.mesh.meshtags(mesh3d, 2, facets_ct.indices, facets_ct.values)
    ds_insulated = ufl.Measure("ds", domain=mesh3d, subdomain_data=surf_meshtags, subdomain_id=insulated_marker)
    ds_left_cc = ufl.Measure('ds', domain=mesh3d, subdomain_data=surf_meshtags, subdomain_id=left_cc_marker)
    ds_right_cc = ufl.Measure('ds', domain=mesh3d, subdomain_data=surf_meshtags, subdomain_id=right_cc_marker)
    ds_active = ufl.Measure('ds', domain=mesh3d, subdomain_data=surf_meshtags, subdomain_id=active_marker)
    ds_inactive = ufl.Measure('ds', domain=mesh3d, subdomain_data=surf_meshtags, subdomain_id=inactive_marker)

    logger.debug('Computing areas..')

    active_area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_active))
    inactive_area2 = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_inactive))
    area_left_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_left_cc))
    area_right_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_right_cc))
    area_insulated = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_insulated))
    total_area = area_left_cc + area_right_cc + area_insulated
    inactive_area = total_area - active_area
    rve_volume = Lx * Ly * Lz
    logger.debug(200 * (inactive_area - inactive_area2)/(inactive_area + inactive_area2))
    logger.info("**************************RESULTS-SUMMARY******************************************")
    logger.info(f"Contact Area @ left cc [sq. um]                          : {area_left_cc:.4e}")
    logger.info(f"Contact Area @ right cc [sq. um]                         : {area_right_cc:.4e}")
    logger.info(f"Insulated Area [sq. um]                                  : {area_insulated:.4e}")
    logger.info(f"Effective Active Material Area [sq. um]                  : {active_area:.4e}")
    logger.info(f"Ineffective Active Material Area [sq. um]                : {inactive_area:.4e}")
    logger.info(f"Total Area [sq. um]                                      : {total_area:.4e}")
    logger.info(f"RVE Volume [cu. um]                                      : {rve_volume:.4e}")
    logger.info(f"Specific Effective Active Material Area [um-1]           : {active_area/rve_volume:.4e}")
    logger.info(f"Specific Ineffective Active Material Area [um-1]         : {inactive_area/rve_volume:.4e}")
    logger.info(f"Time elapsed                                             : {int(timeit.default_timer() - start_time):3.5f}s")
    logger.info("*************************END-OF-SUMMARY*******************************************")
