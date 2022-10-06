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
    parser.add_argument("--loglevel", nargs='?', const=1, default="INFO")

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
    logger = logging.getLogger(f'{data_dir}')
    logger.setLevel(args.loglevel)
    Nx, Ny, Nz = [int(v) for v in grid_info.split("-")]
    Lx = (Nx - 1) * scale_x
    Ly = (Ny - 1) * scale_y
    Lz = (Nz - 1) * scale_z
    tetr_mesh_path = os.path.join(data_dir, 'tetr.xdmf')
    tria_mesh_path = os.path.join(data_dir, 'tria.xdmf')
    output_current_path = os.path.join(data_dir, 'current.xdmf')
    output_potential_path = os.path.join(data_dir, 'potential.xdmf')

    logger.debug("Loading tetrahedra (dim = 3) mesh..")

    with dolfinx.io.XDMFFile(comm, tetr_mesh_path, "r") as infile3:
        mesh3d = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
        ct = infile3.read_meshtags(mesh3d, name="Grid")
    mesh3d.topology.create_connectivity(mesh3d.topology.dim, mesh3d.topology.dim - 1)

    logger.debug("Loading effective electrolyte..")
    with open(os.path.join(data_dir, 'effective_electrolyte.pickle'), 'rb') as handle:
        effective_electrolyte = list(pickle.load(handle))

    # logger.debug("Loading triangles (dim = 2) mesh..")
    # with dolfinx.io.XDMFFile(comm, tria_mesh_path, "r") as infile2:
    #     mesh2d = infile2.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
    #     facets_ct = infile2.read_meshtags(mesh2d, name="Grid")
    # surf_meshtags = dolfinx.mesh.meshtags(mesh3d, 2, facets_ct.indices, facets_ct.values)

    left_cc_marker = constants.surface_tags["left_cc"]
    right_cc_marker = constants.surface_tags["right_cc"]
    insulated_marker = constants.surface_tags["insulated"]
    active_marker = constants.surface_tags["active_area"]
    inactive_marker = constants.surface_tags["inactive_area"]

    # x0facet = np.array(facets_ct.indices[facets_ct.values == left_cc_marker])
    # x1facet = np.array(facets_ct.indices[facets_ct.values == right_cc_marker])
    # insulated_facet = np.array(facets_ct.indices[facets_ct.values == insulated_marker])
    # active_facet = np.array(facets_ct.indices[facets_ct.values == active_marker])
    # inactive_facet = np.array(facets_ct.indices[facets_ct.values == inactive_marker])

    # Dirichlet BCs
    V = dolfinx.fem.FunctionSpace(mesh3d, ("Lagrange", 2))
    u0 = dolfinx.fem.Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(voltage)

    u1 = dolfinx.fem.Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0.0)
    
    x0facet = dolfinx.mesh.locate_entities_boundary(mesh3d, 2,
                                    lambda x: np.isclose(x[1], 0.0))
    x1facet = dolfinx.mesh.locate_entities_boundary(mesh3d, 2,
                                    lambda x: np.isclose(x[1], Ly))
    insulated_facet = dolfinx.mesh.locate_entities_boundary(mesh3d, 2, lambda x: np.logical_and(np.logical_not(np.isclose(x[1], 0)), np.logical_not(np.isclose(x[1], Ly))))

    def is_active_area(x, effective_electrolyte):
        ret_val = np.zeros(x.shape[1])
        for idx in range(x.shape[1]):
            c = tuple(x[:, idx])
            coord = (int(np.rint(c[0]/scale_x)), int(np.rint(c[1]/scale_y)), int(np.rint(c[2]/scale_z)))
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

    x0bc = dolfinx.fem.dirichletbc(u0, dolfinx.fem.locate_dofs_topological(V, 2, x0facet))
    x1bc = dolfinx.fem.dirichletbc(u1, dolfinx.fem.locate_dofs_topological(V, 2, x1facet))
    n = -ufl.FacetNormal(mesh3d)
    ds = ufl.Measure("ds", domain=mesh3d, subdomain_data=surf_meshtags, subdomain_id=insulated_marker)
    ds_left_cc = ufl.Measure('ds', domain=mesh3d, subdomain_data=surf_meshtags, subdomain_id=left_cc_marker)
    ds_right_cc = ufl.Measure('ds', domain=mesh3d, subdomain_data=surf_meshtags, subdomain_id=right_cc_marker)
    ds_active = ufl.Measure('ds', domain=mesh3d, subdomain_data=surf_meshtags2, subdomain_id=active_marker)
    ds_inactive = ufl.Measure('ds', domain=mesh3d, subdomain_data=surf_meshtags2, subdomain_id=inactive_marker)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh3d)

    # bulk conductivity [S.m-1]
    kappa_0 = dolfinx.fem.Constant(mesh3d, PETSc.ScalarType(constants.KAPPA0))
    f = dolfinx.fem.Constant(mesh3d, PETSc.ScalarType(0.0))
    g = dolfinx.fem.Constant(mesh3d, PETSc.ScalarType(0.0))

    a = ufl.inner(kappa_0 * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ds

    options = {
               "ksp_type": "gmres",
               "pc_type": "hypre",
               "ksp_rtol": 1.0e-12
               }

    model = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[x0bc, x1bc], petsc_options=options)

    logger.debug('Solving problem..')
    uh = model.solve()
    
    # Save solution in XDMF format
    with dolfinx.io.XDMFFile(comm, output_potential_path, "w") as outfile:
        outfile.write_mesh(mesh3d)
        outfile.write_function(uh)

    # # Update ghost entries and plot
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    # Post-processing: Compute derivatives
    grad_u = ufl.grad(uh)

    W = dolfinx.fem.FunctionSpace(mesh3d, ("Lagrange", 1))
    current_expr = dolfinx.fem.Expression(kappa_0 * ufl.sqrt(ufl.inner(grad_u, grad_u)), W.element.interpolation_points)
    current_h = dolfinx.fem.Function(W)
    current_h.interpolate(current_expr)

    with dolfinx.io.XDMFFile(comm, output_current_path, "w") as file:
        file.write_mesh(mesh3d)
        file.write_function(current_h)

    insulated_area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds))
    active_area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_active))
    inactive_area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_inactive))
    area_left_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_left_cc))
    area_right_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_right_cc))
    i_left_cc = (1/area_left_cc) * dolfinx.fem.assemble_scalar(dolfinx.fem.form(kappa_0 * ufl.sqrt(ufl.inner(grad_u, grad_u)) * ds_left_cc))
    i_right_cc = (1/area_right_cc) * dolfinx.fem.assemble_scalar(dolfinx.fem.form(kappa_0 * ufl.sqrt(ufl.inner(grad_u, grad_u)) * ds_right_cc))
    i_insulated = (1/insulated_area) * dolfinx.fem.assemble_scalar(dolfinx.fem.form(kappa_0 * ufl.sqrt(ufl.inner(grad_u, grad_u)) * ds))
    volume = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(mesh3d)))
    solution_trace_norm = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(ufl.grad(uh), n) ** 2 * ds)) ** 0.5
    avg_solution_trace_norm = solution_trace_norm / insulated_area
    deviation_in_current = np.around(100 * 2 * np.abs(area_left_cc * i_left_cc - area_right_cc * i_right_cc) / (area_left_cc * i_left_cc + area_right_cc * i_right_cc), 2)
    logger.info("**************************RESULTS-SUMMARY******************************************")
    logger.info("Contact Area @ left cc [sq. um]                 : {:.4e}".format(area_left_cc))
    logger.info("Contact Area @ right cc [sq. um]                : {:.4e}".format(area_right_cc))
    logger.info("Current density @ left cc                       : {:.4e}".format(i_left_cc))
    logger.info("Current density @ right cc                      : {:.4e}".format(i_right_cc))
    logger.info("Effective Active Material Area [sq. um]         : {:.4e}".format(active_area))
    logger.info("Ineffective Active Material Area [sq. um]       : {:.4e}".format(inactive_area))
    logger.info("Insulated Area [sq. um]                         : {:.4e}".format(insulated_area))
    logger.info("Total Area [sq. um]                             : {:.4e}".format(area_left_cc + area_right_cc + insulated_area))
    logger.info("Total Volume [cu. um]                           : {:.4e}".format(volume))
    logger.info("Electrolyte Volume Fraction                     : {:.2%}".format(volume/(Lx * Ly * Lz)))
    logger.info("Effective Active Material Specific Area         : {:.4e}".format(active_area/(Lx * Ly * Lz)))
    logger.info("Bulk conductivity [S.m-1]                       : {:.4e}".format(0.1))
    logger.info("Effective conductivity [S.m-1]                  : {:.4e}".format(Ly * area_left_cc * i_left_cc / (voltage * (Lx * Lz))))
    logger.info(f"Homogeneous Neumann BC trace                    : {solution_trace_norm:.2e}")
    logger.info(f"Area-averaged Homogeneous Neumann BC trace      : {avg_solution_trace_norm:.2e}")
    logger.info("Deviation in current at two current collectors  : {:.2f}%".format(deviation_in_current))
    logger.info(f"Time elapsed                                    : {int(timeit.default_timer() - start_time):3.5f}s")
    logger.info("*************************END-OF-SUMMARY*******************************************")
