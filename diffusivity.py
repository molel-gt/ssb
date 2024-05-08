#!/usr/bin/env python3

import os
import time
import timeit

import argparse
import logging
import numpy as np
import ufl

from dolfinx import cpp, fem, io, mesh, nls, plot
from mpi4py import MPI
from petsc4py import PETSc

import commons, configs, constants, utils

markers = commons.SurfaceMarkers()
phases = commons.Phases()
i_exchange = 1e-4
R = 8.314
F_farad = 96485
T = 298


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Diffusivity')
    parser.add_argument('--grid_extents', help='Nx-Ny-Nz_Ox-Oy-Oz size_location', required=True)
    parser.add_argument('--root_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage", nargs='?', const=1, default=100e-3)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='VOXEL_SCALING4', type=str)
    args = parser.parse_args()
    start_time = timeit.default_timer()
    data_dir = os.path.join(args.root_folder, args.grid_extents)
    loglevel = configs.get_configs()['LOGGING']['level']
    grid_extents = args.grid_extents
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    formatter = logging.Formatter(f'%(levelname)s:%(asctime)s:{grid_extents}:%(message)s')
    fh = logging.FileHandler(os.path.basename(__file__).replace(".py", ".log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    comm = MPI.COMM_WORLD

    scaling = configs.get_configs()[args.scaling]
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])

    Lx, Ly, Lz = [float(v) - 1 for v in grid_extents.split("_")[0].split("-")]
    Lx = Lx * scale_x
    Ly = Ly * scale_y
    Lz = Lz * scale_z

    tria_mesh_path = os.path.join(data_dir, 'tria.xdmf')
    line_mesh_path = os.path.join(data_dir, 'line.xdmf')
    output_current_path = os.path.join(data_dir, 'current.xdmf')
    output_potential_path = os.path.join(data_dir, 'potential.xdmf')
    concentration_path = os.path.join(data_dir, "concentration.xdmf")

    left_cc_marker = markers.left_cc
    right_cc_marker = markers.right_cc
    insulated_marker = markers.insulated

    logger.debug("Loading triangle (dim = 2) mesh..")
    with io.XDMFFile(comm, tria_mesh_path, "r") as infile3:
        domain = infile3.read_mesh(cpp.mesh.GhostMode.none, 'Grid')
        ct = infile3.read_meshtags(domain, name="Grid")
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
    with io.XDMFFile(comm, line_mesh_path, "r") as infile2:
        ft = infile2.read_meshtags(domain, name="Grid")
    meshtags = mesh.meshtags(domain, domain.topology.dim - 1, ft.indices, ft.values)
    domaintags = mesh.meshtags(domain, domain.topology.dim, ct.indices, ct.values)

    # potential problem
    Q = fem.functionspace(domain, ("CG", 2))

    u0 = fem.Function(Q)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(args.voltage)

    u1 = fem.Function(Q)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0.0)

    left_boundary = ft.find(markers.left_cc)
    right_boundary = ft.find(markers.right_cc)
    cells_am = ct.find(phases.active_material)
    cells_se = ct.find(phases.electrolyte)
    left_bc = fem.dirichletbc(u0, fem.locate_dofs_topological(Q, domain.topology.dim - 1, left_boundary))
    right_bc = fem.dirichletbc(u1, fem.locate_dofs_topological(Q, domain.topology.dim - 1, right_boundary))
    n = ufl.FacetNormal(domain)
    x = ufl.SpatialCoordinate(domain)
    h = 2 * ufl.Circumradius(domain)
    dS = ufl.Measure('dS', domain=domain, subdomain_data=meshtags)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=meshtags)
    dx = ufl.Measure('dx', domain=domain, subdomain_data=domaintags)

    # Define variational problem
    u = fem.Function(Q)
    u.name = "potential"
    v = ufl.TestFunction(Q)

    # bulk conductivity [S.m-1]
    # kappa = fem.Function(Q)
    # kappa.x.array[cells_se] = np.full_like(cells_se, constants.KAPPA0, dtype=PETSc.ScalarType)
    # kappa.x.array[cells_am] = np.full_like(cells_am, 0, dtype=PETSc.ScalarType)
    # sigma = fem.Function(Q)
    # sigma.x.array[cells_se] = np.full_like(cells_se, 0, dtype=PETSc.ScalarType)
    # sigma.x.array[cells_am] = np.full_like(cells_am, constants.SIGMA0, dtype=PETSc.ScalarType)

    kappa = fem.Constant(domain, PETSc.ScalarType(constants.KAPPA0))
    sigma = fem.Constant(domain, PETSc.ScalarType(constants.SIGMA0))
    f = fem.Constant(domain, PETSc.ScalarType(0.0))
    g = fem.Constant(domain, PETSc.ScalarType(0.0))

    F = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * dx(phases.electrolyte)
    F += ufl.inner(sigma * ufl.grad(u), ufl.grad(v)) * dx(phases.active_material)
    F -= ufl.inner(f, v) * dx(phases.electrolyte)
    F -= ufl.inner(f, v) * dx(phases.active_material)
    F -= ufl.inner(g, v) * ds(markers.insulated)
    F += ufl.inner(kappa * ufl.grad(u("-")), v("-") * n("-")) * dS(markers.am_se_interface)
    F -= (i_exchange * F_farad / R / T) * ufl.inner(ufl.jump(u, n), ufl.jump(v, n))  * dS(markers.am_se_interface)

    problem = fem.petsc.NonlinearProblem(F, u, bcs=[left_bc, right_bc])
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-12

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    opts['maximum_iterations'] = 100
    ksp.setFromOptions()
    ret = solver.solve(u)

    # Save solution in XDMF format
    with io.XDMFFile(comm, output_potential_path, "w") as outfile:
        outfile.write_mesh(domain)
        outfile.write_function(u)

    logger.debug("Post-process calculations")
    grad_u = ufl.grad(u)
    W = fem.VectorFunctionSpace(domain, ("CG", 1))
    current_expr = fem.Expression(-grad_u, W.element.interpolation_points())
    current_h = fem.Function(W)
    current_h.name = "current_density"
    tol_fun = fem.Function(Q)
    current_h.interpolate(current_expr)

    with io.XDMFFile(comm, output_current_path, "w") as file:
        file.write_mesh(domain)
        file.write_function(current_h)

    logger.debug("Post-process Results Summary")
    insulated_area = domain.comm.reduce(fem.assemble_scalar(fem.form(1 * ds(markers.insulated))), op=MPI.SUM, root=0)
    area_left_cc = domain.comm.reduce(fem.assemble_scalar(fem.form(1 * ds(markers.left_cc))), op=MPI.SUM, root=0)
    area_right_cc = domain.comm.reduce(fem.assemble_scalar(fem.form(1 * ds(markers.right_cc))), op=MPI.SUM, root=0)
    I_left_cc = domain.comm.reduce(fem.assemble_scalar(fem.form(ufl.inner(kappa * current_h, n) * ds(markers.left_cc))), op=MPI.SUM, root=0)
    I_right_cc = domain.comm.reduce(fem.assemble_scalar(fem.form(ufl.inner(sigma * current_h, n) * ds(markers.right_cc))), op=MPI.SUM, root=0)
    I_insulated = domain.comm.reduce(fem.assemble_scalar(fem.form(ufl.inner((kappa + sigma) * current_h, n) * ds)), op=MPI.SUM, root=0)
    volume = domain.comm.reduce(fem.assemble_scalar(fem.form(1 * ufl.dx(domain))), op=MPI.SUM, root=0)
    total_area = area_left_cc + area_right_cc + insulated_area
    error = 100 * 2 * abs(abs(I_left_cc) - abs(I_right_cc)) / (abs(I_left_cc) + abs(I_right_cc))

    if domain.comm.rank == 0:
        print(I_left_cc, I_right_cc)
        logger.info("**************************RESULTS-SUMMARY******************************************")
        logger.info(f"Contact Area @ left cc [sq. um]                 : {area_left_cc * 1e12:.4e}")
        logger.info(f"Contact Area @ right cc [sq. um]                : {area_right_cc * 1e12:.4e}")
        logger.info(f"Insulated Area [sq. um]                         : {insulated_area * 1e12:.4e}")
        logger.info(f"Total Area [sq. um]                             : {total_area * 1e12:.4e}")
        logger.info(f"Current density @ left cc [A/m2]                : {I_left_cc/area_left_cc:.4e}")
        logger.info(f"Current density @ right cc [A/m2]               : {I_right_cc/area_right_cc:.4e}")
        logger.info(f"Current @ left cc [A]                           : {I_left_cc:.4e}")
        logger.info(f"Current @ right cc [A]                          : {I_right_cc:.4e}")
        logger.info(f"Insulated Current [A]                           : {I_insulated:.2e}")
        logger.info(f"Electrolyte Volume [cu. um]                     : {volume * 1e18:.4e}")
        logger.info("Electrolyte Volume Fraction                     : {:.2%}".format(volume / (Lx * Ly * Lz)))
        logger.info(f"Bulk conductivity [S.m-1]                       : {constants.KAPPA0:.4e}")
        logger.info("Effective conductivity [S.m-1]                  : {:.4e}".format(
            Lz * abs(I_left_cc) / (args.voltage * (Lx * Ly))))
        logger.info(f"Conductor Length, L_z [um]                      : {Lz * 1e6:.1e}")
        logger.info(f"Deviation in current at two current collectors  : {error:.2f}%")
        logger.info(f"Voltage                                         : {args.voltage}")
        logger.info(
            f"Time elapsed                                    : {int(timeit.default_timer() - start_time):3.5f}s")
        logger.info("*************************END-OF-SUMMARY*******************************************")
