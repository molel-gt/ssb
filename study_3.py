#!/usr/bin/env python3

import os
import time

import argparse
import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import ufl

from dolfinx import cpp, fem, io, mesh, nls, plot
from mpi4py import MPI
from petsc4py import PETSc

import commons, geometry

import dolfinx

markers = commons.SurfaceMarkers()

# model parameters
SIGMA = 1e-3  # S/m
KAPPA = 1e-1  # S/m
D0 = 1e-13  # m^2/s
F_c = 96485  # C/mol
i0 = 1e-3  # A/m^2
dt = 1e-03  # millisecond
t_iter = 15
theta = 0.5  # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicholson
c_init = 0.01
R = 8.314
T = 298
z = 1
voltage = 0
tau_hat = 5e-6 ** 2 / D0

pulse_iter = 10
i_sup = 1e-6
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
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 2))

    f = dolfinx.fem.Constant(domain, PETSc.ScalarType(0.0))
    g = dolfinx.fem.Constant(domain, PETSc.ScalarType(0.0))
    kappa = dolfinx.fem.Constant(domain, PETSc.ScalarType(KAPPA))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=tags)

    tags = mesh.meshtags(domain, domain.topology.dim - 1, ft.indices, ft.values)

    u, q = ufl.TrialFunction(V), ufl.TestFunction(V)

    F = kappa * ufl.inner(ufl.grad(u), ufl.grad(q)) * ufl.dx 
    F -= ufl.inner(f, q) * ufl.dx
    bcs = []
    F += ufl.inner(g, q) * ds(markers.insulated)
    # s = fem.Constant(domain, PETSc.ScalarType(U_therm))
    r = fem.Constant(domain, PETSc.ScalarType(i0 * z * F_c / (R * T)))
    g_1 = dolfinx.fem.Constant(domain, PETSc.ScalarType(i_sup))
    F += ufl.inner(g_1, q) * ds(markers.left_cc)
    F += r * ufl.inner(phi_m - u - U_therm, q) * ds(markers.right_cc)
    options = {
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "ksp_rtol": 1.0e-14,
            }
    a = ufl.lhs(F)
    L = ufl.rhs(F)

    problem2 = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options=options)
    uh = problem2.solve()
    # write potential to file
    with dolfinx.io.XDMFFile(domain.comm, os.path.join(args.outdir, "potential.xdmf"), "w") as file:
        file.write_mesh(domain)
        file.write_function(uh)
    W = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))
    grad_u = ufl.grad(uh)
    current_expr = dolfinx.fem.Expression(kappa * ufl.sqrt(ufl.inner(grad_u, grad_u)), W.element.interpolation_points())
    current_h = dolfinx.fem.Function(W)
    current_h.interpolate(current_expr)
    # write current to file
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, os.path.join(args.outdir, "current.xdmf"), "w") as file:
        file.write_mesh(domain)
        file.write_function(current_h)
    # compute standard deviation of current
    l_right_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds(markers.right_cc)))
    l_left_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds(markers.left_cc)))
    i_surf_avg = dolfinx.fem.assemble_scalar(dolfinx.fem.form(current_h * ds(markers.right_cc))) / l_right_cc
    i_surf_std = (dolfinx.fem.assemble_scalar(dolfinx.fem.form((current_h - i_surf_avg) ** 2 * ds(markers.right_cc))) / l_right_cc) ** 0.5
    print("Relative Radius: " + args.outdir.split('/')[-1] + ", STD:", i_surf_std / i_surf_avg)
    Wa = KAPPA * R * T / (l_left_cc * F_c * i0)
    # print(f"Wa: {Wa}")
