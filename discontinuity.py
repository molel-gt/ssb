#!/usr/bin/env python3

import os
import time

import argparse

import numpy as np
import ufl

from dolfinx import cpp, fem, io, mesh, nls
from mpi4py import MPI
from petsc4py import PETSc

import commons

markers = commons.SurfaceMarkers()

# model parameters
KAPPA = 1e-1  # [S/m]
faraday_const = 96485  # [C/mol]
# i0 = 1e-2  # [A/m^2]
R = 8.314  # [J/K/mol]
T = 298  # [K]
z = 1
voltage = 0
alpha_a = alpha_c = 0.5
i_sup = 1e0  # [A/m^2]
phi_m = 1  # [V]
U_therm = 0  # [V]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument('--outdir', help='Working directory', required=True)

    args = parser.parse_args()
    work_dir = args.outdir
    comm = MPI.COMM_WORLD
    start = time.time()
    Lx = Ly = 1.0
    with io.XDMFFile(comm, os.path.join(f"{work_dir}", "tria.xdmf"), "r") as xdmf:
        domain = xdmf.read_mesh(cpp.mesh.GhostMode.shared_facet, name="Grid")
        ct = xdmf.read_meshtags(domain, name="Grid")

    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
    with io.XDMFFile(comm, os.path.join(f"{work_dir}", "line.xdmf"), "r") as xdmf:
        ft = xdmf.read_meshtags(domain, name="Grid")
    tags = mesh.meshtags(domain, domain.topology.dim - 1, ft.indices, ft.values)
    left_facets = ft.find(markers.left_cc)
    right_facets = ft.find(markers.right_cc)

    # domain = mesh.create_rectangle(comm, points=((0.0, 0.0), (Lx, Ly)), n=(32, 32), cell_type=mesh.CellType.triangle)
    #
    # left_facets = mesh.locate_entities_boundary(domain, dim=domain.topology.dim - 1,
    #                                             marker=lambda x: np.isclose(x[1], 0.0))
    # right_facets = mesh.locate_entities_boundary(domain, dim=domain.topology.dim - 1,
    #                                             marker=lambda x: np.logical_and(
    #                                                 np.isclose(x[1], Lx), np.less_equal(x[0], 0.5)
    #                                             )
    #                                              )
    f = fem.Constant(domain, PETSc.ScalarType(0))
    g = fem.Constant(domain, PETSc.ScalarType(0))

    kappa = fem.Constant(domain, PETSc.ScalarType(1))
    dx = ufl.Measure('dx', domain=domain)
    ds = ufl.Measure('ds', domain=domain, subdomain_data=tags)

    n = ufl.FacetNormal(domain)
    V = fem.FunctionSpace(domain, ("CG", 2))
    # u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    u = fem.Function(V)
    v = ufl.TestFunction(V)

    u_left = fem.Function(V)
    with u_left.vector.localForm() as u0_loc:
        u0_loc.set(0)
    u_right = fem.Function(V)
    with u_right.vector.localForm() as u1_loc:
        u1_loc.set(1)
    left_bc = fem.dirichletbc(u_left, fem.locate_dofs_topological(V, 1, left_facets))
    right_bc = fem.dirichletbc(u_right, fem.locate_dofs_topological(V, 1, right_facets))
    bcs = [left_bc, right_bc]

    F = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * dx
    F -= ufl.inner(f, v) * dx
    F += ufl.inner(g, v) * ds

    problem = fem.petsc.NonlinearProblem(F, u, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    # solver.convergence_criterion = "residual"
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    solver.max_it = 100

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    opts['monitor_convergence'] = True
    ksp.setFromOptions()
    ret = solver.solve(u)

    # write potential to file
    with io.XDMFFile(domain.comm, os.path.join(args.outdir, "potential.xdmf"), "w") as file:
        file.write_mesh(domain)
        file.write_function(u)

    # compute current density
    W = fem.VectorFunctionSpace(domain, ("CG", 1))

    current_expr = fem.Expression(-kappa * ufl.grad(u), W.element.interpolation_points())
    current_h = fem.Function(W)
    current_h.interpolate(current_expr)

    # write current to file
    with io.XDMFFile(MPI.COMM_WORLD, os.path.join(args.outdir, "current.xdmf"), "w") as file:
        file.write_mesh(domain)
        file.write_function(current_h)

    area_left_cc = fem.assemble_scalar(fem.form(1 * ds(markers.left_cc)))
    area_right_cc = fem.assemble_scalar(fem.form(1 * ds(markers.right_cc)))
    print(area_left_cc, area_right_cc)
    I_left_cc = fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.left_cc)))
    i_left_cc = I_left_cc / area_left_cc
    I_right_cc = fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.right_cc)))
    i_right_cc = I_right_cc / area_right_cc

    print(i_right_cc, i_left_cc)
    print(I_left_cc, I_right_cc)
