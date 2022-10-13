#!/usr/bin/env python3

import os
import pickle
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

    logger.debug("Loading tetrahedra (dim = 3) mesh..")

    with dolfinx.io.XDMFFile(comm, tetr_mesh_path, "r") as infile3:
        mesh3d = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
        ct = infile3.read_meshtags(mesh3d, name="Grid")
    mesh3d.topology.create_connectivity(mesh3d.topology.dim, mesh3d.topology.dim - 1)

    logger.debug("Loading triangles (dim = 2) mesh..")
    with dolfinx.io.XDMFFile(comm, tria_mesh_path, "r") as infile3:
        mesh2d = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
        facets_ct = infile3.read_meshtags(mesh2d, name="Grid")

    logger.debug("Loading effective electrolyte..")
    with open(os.path.join(data_dir, 'effective_electrolyte.pickle'), 'rb') as handle:
        effective_electrolyte = list(pickle.load(handle))

    left_cc_marker = constants.surface_tags["left_cc"]
    right_cc_marker = constants.surface_tags["right_cc"]
    insulated_marker = constants.surface_tags["insulated"]
    active_marker = constants.surface_tags["active_area"]
    inactive_marker = constants.surface_tags["inactive_area"]

    x0facet = np.array(facets_ct.indices[facets_ct.values == left_cc_marker])
    x1facet = np.array(facets_ct.indices[facets_ct.values == right_cc_marker])
    insulated_facet = np.array(facets_ct.indices[facets_ct.values == insulated_marker])
    active_facet = np.array(facets_ct.indices[facets_ct.values == active_marker])
    inactive_facet = np.array(facets_ct.indices[facets_ct.values == inactive_marker])

    # surf_meshtags = dolfinx.mesh.meshtags(mesh2d, 2, facets_ct.indices, facets_ct.values)
    # n = -ufl.FacetNormal(mesh2d)
    # ds_left_cc = ufl.Measure('dx', domain=mesh2d, subdomain_data=surf_meshtags, subdomain_id=left_cc_marker)
    # ds_right_cc = ufl.Measure('dx', domain=mesh2d, subdomain_data=surf_meshtags, subdomain_id=right_cc_marker)
    # ds_insulated = ufl.Measure('dx', domain=mesh2d, subdomain_data=surf_meshtags, subdomain_id=insulated_marker)
    # ds_active = ufl.Measure('dx', domain=mesh2d, subdomain_data=surf_meshtags, subdomain_id=active_marker)
    # ds_inactive = ufl.Measure('dx', domain=mesh2d, subdomain_data=surf_meshtags, subdomain_id=inactive_marker)
    # x = ufl.SpatialCoordinate(mesh2d)

    x = ufl.SpatialCoordinate(mesh3d)

    x0facet = dolfinx.mesh.locate_entities_boundary(mesh3d, 2,
                                    lambda x: np.isclose(x[1], 0.0))
    x1facet = dolfinx.mesh.locate_entities_boundary(mesh3d, 2,
                                    lambda x: np.isclose(x[1], Ly))
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

    facets_ct_indices = np.hstack((x0facet, x1facet, insulated_facet))
    facets_ct_values = np.hstack((np.ones(x0facet.shape[0], dtype=np.int32), right_cc_marker * np.ones(x1facet.shape[0], dtype=np.int32),
                                insulated_marker * np.ones(insulated_facet.shape[0], dtype=np.int32)))
    surf_meshtags = dolfinx.mesh.meshtags(mesh3d, 2, facets_ct_indices, facets_ct_values)
    facets_ct_indices2 = np.hstack((active_facet, inactive_facet))
    facets_ct_values2 = np.hstack((active_marker * np.ones(active_facet.shape[0], dtype=np.int32), inactive_marker * np.ones(inactive_facet.shape[0], dtype=np.int32)))
    surf_meshtags2 = dolfinx.mesh.meshtags(mesh3d, 2, facets_ct_indices2, facets_ct_values2)
    ds_insulated = ufl.Measure("ds", domain=mesh3d, subdomain_data=surf_meshtags, subdomain_id=insulated_marker)
    ds_left_cc = ufl.Measure('ds', domain=mesh3d, subdomain_data=surf_meshtags, subdomain_id=left_cc_marker)
    ds_right_cc = ufl.Measure('ds', domain=mesh3d, subdomain_data=surf_meshtags, subdomain_id=right_cc_marker)
    ds_active = ufl.Measure('ds', domain=mesh3d, subdomain_data=surf_meshtags2, subdomain_id=active_marker)
    ds_inactive = ufl.Measure('ds', domain=mesh3d, subdomain_data=surf_meshtags2, subdomain_id=inactive_marker)

    logger.debug('Computing areas..')

    active_area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_active))
    inactive_area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_inactive))
    area_left_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_left_cc))
    area_right_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_right_cc))
    area_insulated = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_insulated))
    logger.info("**************************RESULTS-SUMMARY******************************************")
    logger.info("Contact Area @ left cc [sq. um]                          : {:.4e}".format(area_left_cc))
    logger.info("Contact Area @ right cc [sq. um]                         : {:.4e}".format(area_right_cc))
    logger.info("Insulated Area [sq. um]                                  : {:.4e}".format(area_insulated))
    logger.info("Effective Active Material Area [sq. um]                  : {:.4e}".format(active_area))
    logger.info("Ineffective Active Material Area [sq. um]                : {:.4e}".format(inactive_area))
    logger.info("Total Area [sq. um]                                      : {:.4e}".format(active_area + inactive_area))
    logger.info("Total Volume [cu. um]                                    : {:.4e}".format(Lx * Ly * Lz))
    logger.info("Specific Effective Active Material Area [sq. um]         : {:.4e}".format(active_area/(Lx * Ly * Lz)))
    logger.info("Specific Ineffective Active Material Area [sq. um]       : {:.4e}".format(inactive_area/(Lx * Ly * Lz)))
    logger.info(f"Time elapsed                                             : {int(timeit.default_timer() - start_time):3.5f}s")
    logger.info("*************************END-OF-SUMMARY*******************************************")
