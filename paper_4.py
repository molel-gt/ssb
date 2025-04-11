#!/usr/bin/env python3
import argparse
import cmath
import json
import os
import sys
import time

import basix
import dolfinx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scifem
import ufl

from dolfinx import fem, mesh

from dolfinx.cpp.mesh import cell_num_entities

from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block, NonlinearProblem
from dolfinx.fem import petsc
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import gmshio, VTXWriter
from dolfinx.nls import petsc as petsc_nls
from mpi4py import MPI
from petsc4py import PETSc
from ufl import avg, div, dot, grad, inner, jump

import commons, mesh_utils, solver_params, solvers, utils

R = 8.314
T = 298
F = 96485
a = 100
kappa = 4
a_a = 0.5
a_c = 0.5

def phi_linear(y, solver):
    return (-solver.V_cell/(solver.d_p * solver.lmbda))/(np.exp(solver.lmbda * solver.L)+np.exp(-solver.lmbda * solver.L)) * (np.exp(solver.lmbda * y) - np.exp(-solver.lmbda * y)) + solver.V_cell * y / solver.d_p


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    w = 1e2
    bv_solver = solvers.ShuntCurrentsSolver(comm, i0=kappa*R*T*1e6**2/(F*a*(a_a+a_c)))
    bv_solver.setup()
    bv_solver.solve_bv()

    # fig, ax = plt.subplots()
    y = np.linspace(0, bv_solver.L, 1001)
    # # y.reshape(-1, 1)
    # PETSc.Sys.Print("Here")
    u_lin = phi_linear(y, bv_solver)
    # PETSc.Sys.Print("Here")
    # u_bv = bv_solver.phi_bv()
    # PETSc.Sys.Print("Here")
    # ax.plot(y, u_lin, "b", label='Linear', linewidth=2)
    # PETSc.Sys.Print("Here 4")
    # ax.plot(y, u_bv, "r", label='Butler-Volmer', linewidth=2)
    # PETSc.Sys.Print("Here 5")
    # # ax.plot([0, 100], [-50, 50], 'k', linewidth=0.5, linestyle='--', label='Electrode potential')
    # ax.grid(color='cyan', linewidth=0.5)
    # ax.set_ylabel(r'$\phi$ [V]')
    # ax.set_xlabel(r'Cell number')
    # # ax.set_xlim([0, bv_solver.N_s])
    # # ax.set_ylim([-50, 50])
    # ax.legend()
    # ax.set_box_aspect(1);
    # ax.minorticks_on();
    # plt.tight_layout()
    # plt.savefig(os.path.join(figures_dir, "potential.eps"), format="eps")
    # plt.show()
    u_lin_expr = fem.Expression(bv_solver.phi_linear(), bv_solver.V.element.interpolation_points)
    u_bv_expr = fem.Expression(bv_solver.phi_bv(), bv_solver.V.element.interpolation_points)
    u_linear = fem.Function(bv_solver.V)
    u_linear.interpolate(u_lin_expr)
    u_butler_volmer = fem.Function(bv_solver.V)
    u_butler_volmer.interpolate(u_bv_expr)

    bb_trees = bb_tree(bv_solver.domain, bv_solver.domain.topology.dim)
    n_points = 10000
    tol = 1e-8  # Avoid hitting the outside of the domain
    x = np.linspace(tol, bv_solver.L - tol, n_points)
    points = np.zeros((3, n_points))
    points[0] = x
    u_values = []
    cells = []
    points_on_proc = []
    cell_candidates = compute_collisions_points(bb_trees, points.T)
    colliding_cells = compute_colliding_cells(bv_solver.domain, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values_lin = u_linear.eval(points_on_proc, cells)
    u_values_bv = bv_solver.u.eval(points_on_proc, cells)
    # u_values_bv = u_butler_volmer.eval(points_on_proc, cells)
    fig, ax = plt.subplots()
    # u_lin = phi_linear(points_on_proc[:, 0], bv_solver)
    # ax.plot(points_on_proc[:, 0], u_values_bv, "b", label='Butler-Volmer', linewidth=2)
    ax.plot(points_on_proc[:, 0], u_values_bv, "b", label='Butler-Volmer', linewidth=2)
    ax.plot(y, u_lin, "r", label='Linear', linewidth=2)
    ax.plot([0, bv_solver.L], [0, bv_solver.N_s/2], 'k', linewidth=0.5, linestyle='--', label='Electrode potential')
    ax.grid(color='cyan', linewidth=0.5)
    ax.set_ylabel(r'$\phi$ [V]')
    ax.set_xlabel(r'Cell number')
    # ax.set_xlim([0, N_s])
    # ax.set_ylim([-50, 50])
    ax.legend()
    ax.set_box_aspect(1);
    ax.minorticks_on();
    plt.tight_layout()
    # plt.savefig(os.path.join(figures_dir, "potential.eps"), format="eps")
    plt.show()
