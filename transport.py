#!/usr/bin/env python3

import os
import timeit

import argparse
import dolfinx
import logging
import numpy as np
import ufl

from dolfinx import cpp, fem, io, mesh
from mpi4py import MPI
from petsc4py import PETSc

import commons, configs, constants

markers = commons.SurfaceMarkers()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument('--grid_extents', help='Nx-Ny-Nz_Ox-Oy-Oz size_location', required=True)
    parser.add_argument('--phase', default=1, type=int, nargs='?', help='0 - VOID, 1 - SE, 2 - AM')
    parser.add_argument("--voltage", help="applied voltage", nargs='?', const=1, default=1)

    args = parser.parse_args()
    data_dir = f'mesh/{args.grid_extents}/{args.phase}'
    voltage = args.voltage
    comm = MPI.COMM_WORLD
    rank = comm.rank
    start_time = timeit.default_timer()
    scaling = configs.get_configs()['VOXEL_SCALING']
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])
    loglevel = configs.get_configs()['LOGGING']['level']

    grid_extents = args.grid_extents
    FORMAT = f'%(asctime)s: %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(f'{data_dir}')
    logger.setLevel(loglevel)
    Lx, Ly, Lz = [float(v) - 1 for v in grid_extents.split("_")[0].split("-")]
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

    logger.debug("Loading tetrahedra (dim = 3) mesh..")
    with io.XDMFFile(comm, tetr_mesh_path, "r") as infile3:
        domain = infile3.read_mesh(cpp.mesh.GhostMode.none, 'Grid')
        ct = infile3.read_meshtags(domain, name="Grid")
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)

    # Dirichlet BCs
    V = fem.FunctionSpace(domain, ("Lagrange", 2))
    u0 = fem.Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(voltage)

    u1 = fem.Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0.0)
    
    x0facet = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[1], 0.0))
    x1facet = mesh.locate_entities_boundary(domain, 2, lambda x: np.isclose(x[1], Ly))
    boundary_surface = mesh.locate_entities_boundary(domain, 2, lambda x: np.isfinite(x[0]))
    insulated_facet = np.asarray(sorted(
        set(boundary_surface).difference(
            set(x0facet).add(set(x1facet))
        )
    )
    )
    # insulated_facet = mesh.locate_entities_boundary(domain, 2, lambda x: np.logical_and(np.logical_not(np.isclose(x[1], 0)), np.logical_not(np.isclose(x[1], Ly))))

    facets_ct_indices = np.hstack((x0facet, x1facet, insulated_facet))
    facets_ct_values = np.hstack(
        (
            markers.left_cc * np.ones(x0facet.shape[0], dtype=np.int32),
            markers.right_cc * np.ones(x1facet.shape[0], dtype=np.int32),
            markers.insulated * np.ones(insulated_facet.shape[0], dtype=np.int32)
        )
                                 )
    facets_ct = commons.Facet(facets_ct_indices, facets_ct_values)
    surf_meshtags = mesh.meshtags(domain, 2, facets_ct.indices, facets_ct.values)

    left_bc = fem.dirichletbc(u0, fem.locate_dofs_topological(V, 2, x0facet))
    right_bc = fem.dirichletbc(u1, fem.locate_dofs_topological(V, 2, x1facet))
    n = ufl.FacetNormal(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=surf_meshtags)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # bulk conductivity [S.m-1]
    kappa = fem.Constant(domain, PETSc.ScalarType(constants.KAPPA0))
    f = fem.Constant(domain, PETSc.ScalarType(0.0))
    g = fem.Constant(domain, PETSc.ScalarType(0.0))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ds(markers.insulated)

    options = {
               "ksp_type": "gmres",
               "pc_type": "hypre",
               "ksp_rtol": 1.0e-12
               }

    model = fem.petsc.LinearProblem(a, L, bcs=[left_bc, right_bc], petsc_options=options)

    logger.debug('Solving problem..')
    uh = model.solve()
    
    # Save solution in XDMF format
    with io.XDMFFile(comm, output_potential_path, "w") as outfile:
        outfile.write_mesh(domain)
        outfile.write_function(uh)

    # # Update ghost entries and plot
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    # Post-processing: Compute derivatives
    grad_u = ufl.grad(uh)

    W = fem.FunctionSpace(domain, ("Lagrange", 1))
    current_expr = fem.Expression(-kappa * grad_u, W.element.interpolation_points())
    current_h = fem.Function(W)
    current_h.interpolate(current_expr)

    with io.XDMFFile(comm, output_current_path, "w") as file:
        file.write_mesh(domain)
        file.write_function(current_h)

    insulated_area = fem.assemble_scalar(fem.form(1 * ds(markers.insulated)))
    area_left_cc = fem.assemble_scalar(fem.form(1 * ds(markers.left_cc)))
    area_right_cc = fem.assemble_scalar(fem.form(1 * ds(markers.right_cc)))
    I_left_cc = fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.left_cc)))
    i_left_cc = I_left_cc / area_left_cc
    I_right_cc = fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.right_cc)))
    i_right_cc = I_right_cc / area_right_cc
    I_insulated = fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds))
    i_insulated = I_insulated / insulated_area
    volume = fem.assemble_scalar(fem.form(1 * ufl.dx(domain)))
    total_area = area_left_cc + area_right_cc + insulated_area
    error = 100 * 2 * abs(abs(I_left_cc) - abs(I_right_cc)) / (abs(I_left_cc) + abs(I_right_cc))
    logger.info("**************************RESULTS-SUMMARY******************************************")
    logger.info(f"Contact Area @ left cc [sq. um]                 : {area_left_cc:.4e}")
    logger.info(f"Contact Area @ right cc [sq. um]                : {area_right_cc:.4e}")
    logger.info(f"Current density @ left cc                       : {i_left_cc:.4e}")
    logger.info(f"Current density @ right cc                      : {i_right_cc:.4e}")
    logger.info(f"Insulated Area [sq. um]                         : {insulated_area:.4e}")
    logger.info(f"Total Area [sq. um]                             : {total_area:.4e}")
    logger.info(f"Electrolyte Volume [cu. um]                     : {volume:.4e}")
    logger.info("Electrolyte Volume Fraction                     : {:.2%}".format(volume / (Lx * Ly * Lz)))
    logger.info(f"Bulk conductivity [S.m-1]                       : {constants.KAPPA0:.4e}")
    logger.info("Effective conductivity [S.m-1]                  : {:.4e}".format(Ly * area_left_cc * i_left_cc / (voltage * (Lx * Lz))))
    logger.info(f"Insulated Current [A] : {I_insulated:.2e}")
    logger.info(f"Deviation in current at two current collectors  : {error:.2f}%")
    logger.info(f"Voltage                                         : {args.voltage}")
    logger.info(f"Time elapsed                                    : {int(timeit.default_timer() - start_time):3.5f}s")
    logger.info("*************************END-OF-SUMMARY*******************************************")
