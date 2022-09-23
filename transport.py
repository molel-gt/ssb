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

import constants, utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run simulation..')
    parser.add_argument('--grid_info', help='Nx-Ny-Nz', required=True)
    parser.add_argument('--data_dir', help='directory with mesh files. output will be saved here', required=True, type=str)
    parser.add_argument("--voltage", nargs='?', const=1, default=1)
    parser.add_argument("--scale_x", nargs='?', const=1, default=1, type=lambda f: np.around(float(f), 8))
    parser.add_argument("--scale_y", nargs='?', const=1, default=1, type=lambda f: np.around(float(f), 8))
    parser.add_argument("--scale_z", nargs='?', const=1, default=1, type=lambda f: np.around(float(f), 8))

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
    output_current_path = os.path.join(data_dir, 'current.xdmf')
    output_potential_path = os.path.join(data_dir, 'potential.xdmf')

    logger.debug("Loading volume (dim = 3) mesh..")

    with dolfinx.io.XDMFFile(comm, tetr_mesh_path, "r") as infile3:
        mesh = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
        ct = infile3.read_meshtags(mesh, name="Grid")
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

    logger.debug("Loading surface (dim = 2) mesh..")
    with dolfinx.io.XDMFFile(comm, tria_mesh_path, "r") as infile3:
        mesh_facets = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
        facets_ct = infile3.read_meshtags(mesh, name="Grid")

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

    # Dirichlet BCs
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 2))
    u0 = dolfinx.fem.Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(voltage)

    u1 = dolfinx.fem.Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0.0)
    x0bc = dolfinx.fem.dirichletbc(u0, dolfinx.fem.locate_dofs_topological(V, 2, x0facet))
    x1bc = dolfinx.fem.dirichletbc(u1, dolfinx.fem.locate_dofs_topological(V, 2, x1facet))

    # Neumann BC
    surf_meshtags = dolfinx.mesh.meshtags(mesh, 2, facets_ct.indices, facets_ct.values)
    n = -ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=surf_meshtags, subdomain_id=insulated_marker)
    ds_left_cc = ufl.Measure('ds', domain=mesh, subdomain_data=surf_meshtags, subdomain_id=left_cc_marker)
    ds_right_cc = ufl.Measure('ds', domain=mesh, subdomain_data=surf_meshtags, subdomain_id=right_cc_marker)
    ds_active = ufl.Measure('ds', domain=mesh, subdomain_data=surf_meshtags, subdomain_id=active_marker)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)

    # bulk conductivity [S.m-1]
    kappa_0 = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.1))
    f = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0))
    g = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0))

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
        outfile.write_mesh(mesh)
        outfile.write_function(uh)

    # # Update ghost entries and plot
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    # Post-processing: Compute derivatives
    grad_u = ufl.grad(uh)

    W = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    current_expr = dolfinx.fem.Expression(kappa_0 * ufl.sqrt(ufl.inner(grad_u, grad_u)), W.element.interpolation_points)
    current_h = dolfinx.fem.Function(W)
    current_h.interpolate(current_expr)

    with dolfinx.io.XDMFFile(comm, output_current_path, "w") as file:
        file.write_mesh(mesh)
        file.write_function(current_h)

    insulated_area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds))
    active_area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_active))
    area_left_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_left_cc))
    area_right_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds_right_cc))
    i_left_cc = (1/area_left_cc) * dolfinx.fem.assemble_scalar(dolfinx.fem.form(kappa_0 * ufl.sqrt(ufl.inner(grad_u, grad_u)) * ds_left_cc))
    i_right_cc = (1/area_right_cc) * dolfinx.fem.assemble_scalar(dolfinx.fem.form(kappa_0 * ufl.sqrt(ufl.inner(grad_u, grad_u)) * ds_right_cc))
    i_insulated = (1/insulated_area) * dolfinx.fem.assemble_scalar(dolfinx.fem.form(kappa_0 * ufl.sqrt(ufl.inner(grad_u, grad_u)) * ds))
    total_volume = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(mesh)))
    solution_trace_norm = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(ufl.grad(uh), n) ** 2 * ds)) ** 0.5
    avg_solution_trace_norm = solution_trace_norm / insulated_area
    deviation_in_current = np.around(100 * 2 * np.abs(area_left_cc * i_left_cc - area_right_cc * i_right_cc) / (area_left_cc * i_left_cc + area_right_cc * i_right_cc), 2)
    logger.info("**************************RESULTS-SUMMARY******************************************")
    logger.info("Contact Area @ left cc [sq. um]                 : {:0.4f}".format(area_left_cc))
    logger.info("Contact Area @ right cc [sq. um]                : {:0.4f}".format(area_right_cc))
    logger.info("Current density @ left cc                       : {:.6f}".format(i_left_cc))
    logger.info("Current density @ right cc                      : {:.6f}".format(i_right_cc))
    logger.info("Insulated Area [sq. um]                         : {:,}".format(int(insulated_area)))
    logger.info("Total Area [sq. um]                             : {:,}".format(int(area_left_cc + area_right_cc + insulated_area)))
    logger.info("Total Volume [cu. um]                           : {:,}".format(int(total_volume)))
    logger.info("Electrolyte Volume Fraction                     : {:0.4f}".format(total_volume/(Lx * Ly * Lz)))
    logger.info("Effective Active Material Specific Area         : {:.4f}".format(active_area/(Lx * Ly * Lz)))
    logger.info("Bulk conductivity [S.m-1]                       : {:.4f}".format(0.1))
    logger.info("Effective conductivity [S.m-1]                  : {:.4f}".format(Ly * area_left_cc * i_left_cc / (voltage * (Lx * Lz))))
    logger.info(f"Homogeneous Neumann BC trace                    : {solution_trace_norm:.2e}")
    logger.info(f"Area-averaged Homogeneous Neumann BC trace      : {avg_solution_trace_norm:.2e}")
    logger.info("Deviation in current at two current collectors  : {:.2f}%".format(deviation_in_current))
    logger.info(f"Time elapsed                                    : {int(timeit.default_timer() - start_time):3.5f}s")
    logger.info("*************************END-OF-SUMMARY*******************************************")
