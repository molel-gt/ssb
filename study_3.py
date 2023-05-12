#!/usr/bin/env python3

import os
import time

import argparse

import numpy as np
import ufl

from dolfinx import cpp, fem, io, mesh
from mpi4py import MPI
from petsc4py import PETSc

import commons

markers = commons.SurfaceMarkers()

# model parameters
KAPPA = 1e-1  # [S/m]
F_c = 96485  # [C/mol]
i0 = 1e-3  # [A/m^2]
R = 8.314  # [J/K/mol]
T = 298  # [K]
z = 1
voltage = 0

i_sup = 1e-2
phi_m = 0
U_therm = 0


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
    r = fem.Constant(domain, PETSc.ScalarType(i0 * z * F_c / (R * T)))
    kappa = fem.Constant(domain, PETSc.ScalarType(KAPPA))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=tags)
    dx = ufl.Measure('dx', domain=domain)
    x = ufl.SpatialCoordinate(domain)
    n = ufl.FacetNormal(domain)

    V = fem.FunctionSpace(domain, ("Lagrange", 2))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    F = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    F -= ufl.inner(f, v) * dx
    F += ufl.inner(g, v) * ds(markers.insulated)
    F += ufl.inner(g_1, v) * ds(markers.left_cc)
    F += r * ufl.inner(phi_m - u - U_therm, v) * ds(markers.right_cc)
    options = {
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "ksp_rtol": 1e-12,
            }
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    bcs = []
    problem2 = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options=options)
    uh = problem2.solve()

    # write potential to file
    with io.XDMFFile(domain.comm, os.path.join(args.outdir, "potential.xdmf"), "w") as file:
        file.write_mesh(domain)
        file.write_function(uh)

    # compute current density
    W = fem.VectorFunctionSpace(domain, ("Lagrange", 1))
    grad_u = ufl.grad(uh)
    current_expr = fem.Expression(-kappa * grad_u, W.element.interpolation_points())
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
    Wa = KAPPA * R * T / (l_left_cc * F_c * i0)
    print("Relative Radius: " + args.outdir.split('/')[-1] + ", STD:", i_surf_std / np.abs(i_surf_avg), "current left:", I_left, "current right:", I_right, "Wa:", Wa)
