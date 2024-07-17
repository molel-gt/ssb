#!/usr/bin/env python
# coding: utf-8
import argparse
import itertools
import json
import math
import os
import timeit

import dolfinx
import gmsh
import h5py
import logging
import matplotlib.pyplot as plt
import meshio
import numpy as np
import subprocess
import ufl
import warnings

from dolfinx import cpp, default_real_type, default_scalar_type, fem, io, la, mesh, nls, plot
from dolfinx.fem import petsc
from dolfinx.io import gmshio, VTXWriter, XDMFFile
from dolfinx.nls import petsc as petsc_nls
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from IPython.display import Image

from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 dot, div, dx, ds, dS, grad, inner, grad, avg, jump)

import commons, configs, geometry, utils

warnings.simplefilter('ignore')


kappa_elec = 0.1
kappa_pos_am = 0.1
faraday_constant = 96485
R = 8.3145
T = 298


if __name__ == '__main__':
    data_dir = "output/3dcc/50-250-0/5/4.0/1.0e-06/"
    voltage = 1
    Wa = 1e-3
    markers = commons.Markers()
    comm = MPI.COMM_WORLD
    start_time = timeit.default_timer()
    scaling = configs.get_configs()['MICRON_TO_METER']
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])
    loglevel = configs.get_configs()['LOGGING']['level']
    dimensions = '150-40-0'
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    formatter = logging.Formatter(f'%(levelname)s:%(asctime)s:{data_dir}:{dimensions}:%(message)s')
    fh = logging.FileHandler('3dcc.log')#os.path.basename(__file__).replace(".py", ".log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.debug(data_dir)

    Lx, Ly, Lz = [float(v) for v in dimensions.split("-")]
    Lx = Lx * scale_x
    Ly = Ly * scale_y
    Lz = Lz * scale_z
    results_dir = data_dir #os.path.join(data_dir, f"{args.Wa}")
    utils.make_dir_if_missing(results_dir)
    output_meshfile_path = os.path.join(data_dir, 'mesh.msh')
    output_current_path = os.path.join(results_dir, 'current.bp')
    output_potential_path = os.path.join(results_dir, 'potential.bp')
    frequency_path = os.path.join(results_dir, 'frequency.csv')
    simulation_metafile = os.path.join(results_dir, 'simulation.json')
    left_values_path = os.path.join(results_dir, 'left_values')
    right_values_path = os.path.join(results_dir, 'right_values')

    left_cc_marker = markers.negative_am_v_electrolyte
    right_cc_marker = markers.right
    insulated_marker = markers.insulated_electrolyte

    logger.debug("Loading mesh..")
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = gmshio.read_from_msh(output_meshfile_path, comm, partitioner=partitioner)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)
    logger.debug("Here")
    left_boundary = ft.find(markers.negative_am_v_electrolyte)
    right_boundary = ft.find(markers.right)
    logger.debug("done\n")

    # Dirichlet BCs
    V = fem.functionspace(domain, ("CG", 2))

    n = ufl.FacetNormal(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)

    # Define variational problem
    u = fem.Function(V, name='potential')
    v = ufl.TestFunction(V)

    # bulk conductivity [S.m-1]
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_elec))
    f = fem.Constant(domain, PETSc.ScalarType(0.0))
    g = fem.Constant(domain, PETSc.ScalarType(0.0))

    # i0 = kappa_elec * R * T / (Lz * Wa * faraday_constant)
    A0 = Lx * Ly

    u0 = fem.Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(0)
    left_bc = fem.dirichletbc(u0, fem.locate_dofs_topological(V, 1, left_boundary))

    u1 = fem.Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(voltage)
    right_bc = fem.dirichletbc(u1, fem.locate_dofs_topological(V, 1, right_boundary))

    F = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    F -= ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ds(insulated_marker)
    logger.debug(f'Solving problem..')
    problem = petsc.NonlinearProblem(F, u, bcs=[left_bc, right_bc])
    solver = petsc_nls.NewtonSolver(comm, problem)
    solver.convergence_criterion = "residual"
    solver.maximum_iterations = 100
    solver.atol = np.finfo(float).eps
    solver.rtol = np.finfo(float).eps * 10

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "gmres"
    opts[f"{option_prefix}pc_type"] = "hypre"
    ksp.setFromOptions()
    n_iters, converged = solver.solve(u)
    logger.info(f"Converged in {n_iters} iterations")
    u.x.scatter_forward()

    with VTXWriter(comm, output_potential_path, [u], engine="BP4") as vtx:
        vtx.write(0.0)

    logger.debug("Post-process calculations")
    W = fem.functionspace(domain, ("CG", 1, (3,)))
    current_expr = fem.Expression(-kappa * ufl.grad(u), W.element.interpolation_points())
    current_h = fem.Function(W, name='current_density')
    tol_fun = fem.Function(V)
    tol_fun_left = fem.Function(V)
    tol_fun_right = fem.Function(V)
    current_h.interpolate(current_expr)

    with VTXWriter(comm, output_current_path, [current_h], engine="BP4") as vtx:
        vtx.write(0.0)