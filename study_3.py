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
i0 = 1e1  # [A/m^2]
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
    
    with io.XDMFFile(comm, os.path.join(f"{work_dir}", "tria.xdmf"), "r") as xdmf:
        domain = xdmf.read_mesh(cpp.mesh.GhostMode.shared_facet, name="Grid")
        ct = xdmf.read_meshtags(domain, name="Grid")

    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
    with io.XDMFFile(comm, os.path.join(f"{work_dir}", "line.xdmf"), "r") as xdmf:
        ft = xdmf.read_meshtags(domain, name="Grid")
    tags = mesh.meshtags(domain, domain.topology.dim - 1, ft.indices, ft.values)
    left_cc = ft.find(markers.left_cc)
    right_cc = ft.find(markers.right_cc)

    f = fem.Constant(domain, PETSc.ScalarType(0))
    g = fem.Constant(domain, PETSc.ScalarType(0))
    g_1 = fem.Constant(domain, PETSc.ScalarType(i_sup))
    r = fem.Constant(domain, PETSc.ScalarType(i0 * z * faraday_const / (R * T)))
    kappa = fem.Constant(domain, PETSc.ScalarType(KAPPA))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=tags)
    dx = ufl.Measure('dx', domain=domain)
    x = ufl.SpatialCoordinate(domain)
    n = ufl.FacetNormal(domain)

    V = fem.FunctionSpace(domain, ("Lagrange", 2))
    # u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    v = ufl.TestFunction(V)
    u = fem.Function(V)

    # u_right = fem.Function(V)
    # with u_right.vector.localForm() as u1_loc:
    #     u1_loc.set(0)

    # right_facet = ft.find(markers.right_cc)
    # right_bc = fem.dirichletbc(u_right, fem.locate_dofs_topological(V, 1, right_facet))
    bcs = []

    F = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * dx
    F -= ufl.inner(f, v) * dx
    F += ufl.inner(g, v) * ds(markers.insulated)
    F += ufl.inner(g_1, v) * ds(markers.left_cc)
    # i_bv = i0 * ufl.sinh(0.5 * faraday_const * (phi_m - u - U_therm) / R / T)
    i_bv = i0 * (ufl.exp(alpha_a * faraday_const * (phi_m - u - U_therm) / R / T) - ufl.exp(-alpha_c * faraday_const * (phi_m - u - U_therm) / R / T))
    F += ufl.inner(i_bv, v) * ds(markers.right_cc)

    problem = fem.petsc.NonlinearProblem(F, u, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "residual"
    solver.rtol = 1e-6

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    ksp.setFromOptions()
    ret = solver.solve(u)
    print(f"num iterations: {ret[0]}")

    # write potential to file
    with io.XDMFFile(domain.comm, os.path.join(args.outdir, "potential.xdmf"), "w") as file:
        file.write_mesh(domain)
        file.write_function(u)

    # compute current density
    W = fem.VectorFunctionSpace(domain, ("Lagrange", 1))

    current_expr = fem.Expression(-kappa * ufl.grad(u), W.element.interpolation_points())
    current_h = fem.Function(W)
    current_h.interpolate(current_expr)

    # write current to file
    with io.XDMFFile(MPI.COMM_WORLD, os.path.join(args.outdir, "current.xdmf"), "w") as file:
        file.write_mesh(domain)
        file.write_function(current_h)

    # compute standard deviation of current
    l_right_cc = fem.assemble_scalar(fem.form(1 * ds(markers.right_cc)))
    l_left_cc = fem.assemble_scalar(fem.form(1 * ds(markers.left_cc)))
    I_right = fem.assemble_scalar(fem.form(ufl.inner(n, current_h) * ds(markers.right_cc)))
    I_left = fem.assemble_scalar(fem.form(ufl.inner(n, current_h) * ds(markers.left_cc)))
    i_surf_avg = fem.assemble_scalar(fem.form(ufl.inner(n, current_h) * ds(markers.right_cc))) / l_right_cc
    i_surf_std = (fem.assemble_scalar(fem.form((ufl.inner(n, current_h) - i_surf_avg) ** 2 * ds(markers.right_cc))) / l_right_cc) ** 0.5
    Wa = KAPPA * R * T / (l_left_cc * faraday_const * i0)
    std_norm = i_surf_std / np.abs(i_surf_avg)
    rel_scale = args.outdir.split('/')[-1]
    error = 2 * 100 * abs(abs(I_left) - abs(I_right)) / (abs(I_left) + abs(I_right))
    print(f"relative radius: {rel_scale},", f"Wa: {Wa},", f"norm stdev: {std_norm:.2f},",
          f"current left: {I_left:.2e},", f"current right: {I_right:.2e},", f"error: {error:.2f}%,",
          f"time: {int(time.time() - start):,}s")
