#!/usr/bin/env python3

import os
import timeit

import argparse
import dolfinx
import logging
import numpy as np
import ufl

from dolfinx import mesh, fem, io, nls, log
from mpi4py import MPI
from petsc4py import PETSc

import commons, configs, constants


markers = commons.SurfaceMarkers()
phases = commons.Phases()
# Some constants
D_am = 5e-15
D_se = 0
# electronic conductivity
sigma_am = 1e-1
sigma_se = 1e-3
# ionic conductivity
kappa_am = 0
kappa_se = 0.1

i0 = 1e-2  # exchange current density
phi2 = 0.495
F_c = 96485  # Faraday constant
R = 8.314
T = 298
alpha_a = 0.5
alpha_c = 0.5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Laminate Cell')
    parser.add_argument('--data_dir', help='Directory with tria.xdmf and tetr.xmf mesh files. Output files potential.xdmf and current.xdmf will be saved here', required=True, type=str)

    args = parser.parse_args()
    data_dir = args.data_dir
    # voltage = args.voltage
    comm = MPI.COMM_WORLD
    rank = comm.rank
    start_time = timeit.default_timer()
    loglevel = configs.get_configs()['LOGGING']['level']

    FORMAT = f'%(asctime)s: %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(f'{data_dir}')
    logger.setLevel(loglevel)
    line_mesh_path = os.path.join(data_dir, 'line.xdmf')
    tria_mesh_path = os.path.join(data_dir, 'tria.xdmf')
    output_current_path = os.path.join(data_dir, 'current.xdmf')
    output_potential_path = os.path.join(data_dir, 'potential.xdmf')

    left_cc_marker = markers.left_cc
    right_cc_marker = markers.right_cc
    insulated_marker = markers.insulated

    with io.XDMFFile(MPI.COMM_WORLD, tria_mesh_path, "r") as xdmf:
        mesh2d = xdmf.read_mesh(dolfinx.cpp.mesh.GhostMode.none, name="Grid")
        ct = xdmf.read_meshtags(mesh2d, name="Grid")

    mesh2d.topology.create_connectivity(mesh2d.topology.dim, mesh2d.topology.dim - 1)
    with io.XDMFFile(MPI.COMM_WORLD, line_mesh_path, "r") as xdmf:
        ft = xdmf.read_meshtags(mesh2d, name="Grid")

    Q = fem.FunctionSpace(mesh2d, ("DG", 0))
    kappa = fem.Function(Q)
    sigma = fem.Function(Q)
    d_eff = fem.Function(Q)
    se_cells = ct.find(phases.electrolyte)
    am_cells = ct.find(phases.active_material)
    kappa.x.array[am_cells] = np.full_like(am_cells, kappa_am, dtype=PETSc.ScalarType)
    kappa.x.array[se_cells]  = np.full_like(se_cells, kappa_se, dtype=PETSc.ScalarType)
    sigma.x.array[am_cells] = np.full_like(am_cells, sigma_am, dtype=PETSc.ScalarType)
    sigma.x.array[se_cells]  = np.full_like(se_cells, sigma_se, dtype=PETSc.ScalarType)
    d_eff.x.array[am_cells] = np.full_like(am_cells, D_am, dtype=PETSc.ScalarType)
    d_eff.x.array[se_cells]  = np.full_like(se_cells, D_se, dtype=PETSc.ScalarType)
    kappa = fem.Constant(mesh2d, PETSc.ScalarType(constants.KAPPA0))
    x = ufl.SpatialCoordinate(mesh2d)
    # additions
    f = fem.Constant(mesh2d, PETSc.ScalarType(0.0))
    n = ufl.FacetNormal(mesh2d)
    g = fem.Constant(mesh2d, PETSc.ScalarType(0.0))

    V = fem.FunctionSpace(mesh2d, ("Lagrange", 2))
    v = ufl.TestFunction(V)
    u = fem.Function(V)
    # def i_butler_volmer(phi1=u, phi2=phi2):
    #     return i0  * (ufl.exp(alpha_a * F_c * (phi1 - phi2) / R / T) - ufl.exp(-alpha_c * F_c * (phi1 - phi2) / R / T))

    # left_cc_curr = -i_butler_volmer() / kappa

    fdim = mesh2d.topology.dim - 1
    facet_tag = mesh.meshtags(mesh2d, fdim, ft.indices, ft.values)

    ds = ufl.Measure("ds", domain=mesh2d, subdomain_data=facet_tag)
    u_left_cc = fem.Function(V)
    with u_left_cc.vector.localForm() as u0_loc:
        u0_loc.set(0.5)
    
    u_right_cc = fem.Function(V)
    with u_right_cc.vector.localForm() as u0_loc:
        u0_loc.set(3.7)

    left_cc_facet = ft.find(markers.left_cc)
    left_cc_dofs = fem.locate_dofs_topological(V, 1, left_cc_facet)
    left_cc = fem.dirichletbc(u_left_cc, fem.locate_dofs_topological(V, 1, left_cc_facet))
    right_cc_facet = ft.find(markers.right_cc)
    right_cc_dofs = fem.locate_dofs_topological(V, 1, right_cc_facet)
    right_cc = fem.dirichletbc(u_right_cc, fem.locate_dofs_topological(V, 1, right_cc_facet))

    F = sigma * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx - ufl.inner(g, v) * ds(markers.insulated)
    problem = fem.petsc.NonlinearProblem(F, u, bcs=[left_cc, right_cc])
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-12
    solver.maximum_iterations = 100
    solver.report = True

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "cg"
    opts[f"{option_prefix}pc_type"] = "gamg"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    opts[f"{option_prefix}maximum_iterations"] = 100
    ksp.setFromOptions()

    log.set_log_level(log.LogLevel.WARNING)
    n, converged = solver.solve(u)
    print(f"Number of interations: {n:d}")

    with io.XDMFFile(comm, output_potential_path, "w") as file:
        file.write_mesh(mesh2d)
        file.write_function(u)

    grad_u = ufl.grad(u)
    area_left_cc = fem.assemble_scalar(fem.form(1 * ds(markers.left_cc)))
    area_right_cc = fem.assemble_scalar(fem.form(1 * ds(markers.right_cc)))
    i_left_cc = (1/area_left_cc) * fem.assemble_scalar(fem.form(kappa * ufl.sqrt(ufl.inner(grad_u, grad_u)) * ds(markers.left_cc)))
    i_right_cc = (1/area_right_cc) * fem.assemble_scalar(fem.form(sigma * ufl.sqrt(ufl.inner(grad_u, grad_u)) * ds(markers.right_cc)))

    W = fem.FunctionSpace(mesh2d, ("Lagrange", 1))
    current_expr = fem.Expression(kappa * ufl.sqrt(ufl.inner(grad_u, grad_u)), W.element.interpolation_points())
    current_h = fem.Function(W)
    current_h.interpolate(current_expr)

    with io.XDMFFile(comm, output_current_path, "w") as file:
        file.write_mesh(mesh2d)
        file.write_function(current_h)

    print("Current density @ left cc                       : {:.4e}".format(i_left_cc))
    print("Current density @ right cc                      : {:.4e}".format(i_right_cc))
