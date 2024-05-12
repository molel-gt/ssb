#!/usr/bin/env python3

import argparse
import json
import math
import os
import timeit

import dolfinx
import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
import pyvista
import pyvista as pv
import pyvistaqt as pvqt
import ufl
import warnings

from dolfinx import cpp, default_real_type, default_scalar_type, fem, io, la, mesh, nls, plot
from dolfinx.fem import petsc
from dolfinx.io import gmshio, VTXWriter
from dolfinx.nls import petsc as petsc_nls
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from IPython.display import Image

from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 dot, div, dx, grad, inner, grad, avg, jump)

import commons, configs, geometry, utils, nernst_planck_geo


warnings.simplefilter('ignore')

D_n = 1e-5  # [m2/s]
D_p = 1e-5  # [m2/s]
faraday_const = 96485  # [C/mol]
R = 8.31446  # [J/K/mol]
# kB = 1.380649e-23  # [J/K]
T = 298  # [K]
# e = 1.602176634e-19  # [C]
c_init = 5e3  # [mol/m3]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nernst Planck Equation')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="nernst_planck")
    # parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    # parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1.0, type=float)
    parser.add_argument("--Wa", help="Wagna number: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e3, type=float)
    # parser.add_argument("--gamma", help="interior penalty parameter", nargs='?', const=1, default=15, type=float)
    # parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
    #                     const=1, default='MICRON_TO_METER', type=str)

    args = parser.parse_args()
    markers = nernst_planck_geo.Boundaries()
    output_meshfile_path = os.path.join(args.mesh_folder, 'mesh.msh')
    results_dir = os.path.join(args.mesh_folder, f'args.Wa')
    output_c_n_path = os.path.join(results_dir, 'c_n.bp')
    output_c_p_path = os.path.join(results_dir, 'c_p.bp')
    output_potential_path = os.path.join(results_dir, 'potential.bp')
    simulation_metafile = os.path.join(results_dir, 'simulation.json')
    start_time = timeit.default_timer()

    comm = MPI.COMM_WORLD
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = gmshio.read_from_msh(output_meshfile_path, comm, partitioner=partitioner)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    n = ufl.FacetNormal(domain)
    left_boundary = ft.find(markers.left)
    right_boundary = ft.find(markers.right)

    V = dolfinx.fem.functionspace(domain, ("CG", 1, (3, ), ))

    u_t = dolfinx.fem.Function(V)  # previous step.
    u = dolfinx.fem.Function(V)  # current step
    c_n_t, c_p_t, phi_t = ufl.split(u_t)
    c_n, c_p, phi = ufl.split(u)
    v_p, v_n, v_phi = ufl.TestFunction(V)
    z_n = -1
    z_p = 1

    # initial conditions
    u_t.sub(0).interpolate(lambda x: c_init * np.ones(x[0].shape))
    u_t.sub(1).interpolate(lambda x: c_init * np.ones(x[0].shape))
    u_t.sub(2).interpolate(lambda x: 10e-3 * x[0])  # potential gradient

    # set potential boundary conditions
    ul = fem.Function(V.sub(2).collapse()[0])
    with ul.vector.localForm() as ul_loc:
        ul_loc.set(0)
    left_bc = fem.dirichletbc(ul, fem.locate_dofs_topological(V.sub(2), 1, left_boundary))

    ur = fem.Function(V.sub(2).collapse()[0])
    with ur.vector.localForm() as ur_loc:
        ur_loc.set(10e-3)
    right_bc = fem.dirichletbc(ur, fem.locate_dofs_topological(V.sub(2), 1, right_boundary))

    dt = 1e-3  # time step

    f = fem.Constant(domain, PETSc.ScalarType(0.0))
    g = fem.Constant(domain, PETSc.ScalarType(0.0))

    F_c_n = inner(c_n - c_n_t, v_n) * dx + dt * D_n * inner(grad(c_n), grad(v_n)) * dx
    F_c_n += dt * (inner(D_n * z_n * faraday_const / (R * T) * c_n * grad(phi), grad(v_n)) * dx - inner(D_n * z_n * faraday_const / (R * T) * c_n * grad(phi), n) * v_n * ds)
    # no outflux of negative particles
    F_c_n -= dt * (inner(g, v_n) * ds(markers.left) + inner(g, v_n) * ds(markers.bottom) + inner(g, v_n) * ds(markers.right) + inner(g, v_n) * ds(markers.top))
    
    F_c_p = inner(c_p - c_p_t, v_p) * dx + dt * D_p * inner(grad(c_p), grad(v_p)) * dx
    F_c_p += dt * (inner(D_p * z_p * faraday_const / (R * T) * c_p * grad(phi), grad(v_p)) * dx - inner(D_p * z_p * faraday_const / (R * T) * c_p * grad(phi), n) * v_p * ds)
    F_c_p -= dt * (inner(g, v_p) * ds(markers.bottom) + inner(g, v_p) * ds(markers.top))
    F_c_p -= dt * z_p / faraday_const * (inner(grad(phi), n) * v_p * ds(markers.left) + inner(grad(phi), n) * v_p * ds(markers.right))
    
    F_phi = inner(grad(phi), grad(v_phi)) * dx - inner(g, v_phi) * ds(markers.bottom) - inner(g, v_phi) * ds(markers.top)

    F = F_c_n + F_c_p + F_phi

    problem = petsc.NonlinearProblem(F, u, bcs=[left_bc, right_bc])

    solver = petsc_nls.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    # solver.rtol = 1e-6

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "gmres"
    opts[f"{option_prefix}pc_type"] = "lu"
    # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    T = 50 * dt
    t = 0
    vtx_c_n = VTXWriter(comm, output_c_n_path, [u.sub(0)], engine="BP4")
    vtx_c_p = VTXWriter(comm, output_c_p_path, [u.sub(1)], engine="BP4")
    vtx_phi = VTXWriter(comm, output_potential_path, [u.sub(2)], engine="BP4")
    
    while t < T:
        t += dt
        n, converged = solver.solve(u)
        vtx_c_n.write(t)
        vtx_c_p.write(t)
        vtx_phi.write(t)
        c_n_t = cn
    # with VTXWriter(comm, output_c_n_path, [u.sub(0)], engine="BP4") as vtx:
    #     vtx.write(0.0)
    # with VTXWriter(comm, output_c_p_path, [u.sub(1)], engine="BP4") as vtx:
    #     vtx.write(0.0)

    # with VTXWriter(comm, output_potential_path, [u.sub(2)], engine="BP4") as vtx:
    #     vtx.write(0.0)
