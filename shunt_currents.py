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


params = {
    'figure.figsize': (5, 4.5),
    'font.size' : 12,
    'font.family': 'monospace',
    'axes.labelsize': 12,
    'lines.linewidth': 1.0,
    'legend.fontsize': 12,
    'xtick.direction': "in",
    'xtick.top': True,
    'xtick.minor.visible': True,
    'ytick.direction': "in",
    'ytick.minor.visible': True,
    'ytick.right': True,
    'savefig.format': 'eps',
}
plt.rcParams.update(params)


faraday_const = 96485
R = 8.3145
T = 298

# a = 100
# i0 = 10
kappa = 4

alpha = 0.5
# L_p = 0.02
H_p = 0.03
A_m = 0.006
V_cell = 1.0
d_p = 0.01
V0 = 1e-8
# N_s = 100
# L = N_s * d_p/2
# R_p = L_p / kappa

# w = 99

dtype = PETSc.ScalarType


def R_p(L_p, kappa):
    return L_p / kappa


def left_boundary(x):
    return np.isclose(x[0], 0)


def right_boundary(x):
    return np.isclose(x[0], 1)


def i_port(V, i0_prime):
    return i0_prime * ufl.sqrt(-2 + ufl.exp(alpha * faraday_const * V / R / T) + ufl.exp(-alpha * faraday_const * V / R / T))


def m(V, i0_prime):
    return i0_prime ** 2 * (alpha * faraday_const / R / T) * ufl.sinh(alpha * faraday_const / R / T) / i_port(V=V, i0_prime=i0_prime)


def lambda_squared(V, kappa, L_p, i0_prime):
    return H_p / (kappa * A_m) * m(V=V, i0_prime=i0_prime) / (m(V=V, i0_prime=i0_prime) * R_p(L_p, kappa) - 1)


def f(y, V, i0_prime, L_p, kappa):
    return lambda_squared(V=V, i0_prime=i0_prime, L_p=L_p, kappa=kappa) * (V_cell / d_p * y - i_port(V=V, i0_prime=i0_prime)/m(V=V, i0_prime=i0_prime) - V)


# def I_m(u, kappa=4):
#     return -kappa * A_m * ufl.grad(u)


def I_ds(I_m, dx, L):
    return 1 / L * fem.assemble_scalar(fem.form(I_m * dx))


def run_model(u0, u, v, dx, bcs, N_s, kappa, L_p, i0_prime, tol=1e-8, max_its=10):
    it = 0
    error = tol + 1
    x_fun = fem.Function(u.function_space)
    x_fun.interpolate(lambda x: x[0])
    
    while error > tol and it < max_its:
        F0 = -inner(kappa * grad(u), grad(v)) * dx  - (lambda_squared(V=u0, L_p=L_p, kappa=kappa, i0_prime=i0_prime) * u - f(x[0], V=u0, L_p=L_p, kappa=kappa, i0_prime=i0_prime)) * v * dx
        F = [fem.form(F0)]
        j00 = fem.form(ufl.derivative(F0, u))
        J = [[j00]]
        opts = {
                    'ksp_type': 'fgmres',
                    'pc_type': 'hypre',
                    }
        
        solver = scifem.NewtonSolver(F, J, [u], bcs=bcs, petsc_options=opts)
        
        t0 = time.time()
        solver.solve()
        t1 = time.time()
        error = np.sqrt(fem.assemble_scalar(fem.form(((x[0] - 2 * u/N_s - u0)) ** 2 * dx)))
        u0.x.array[:] = x_fun.x.array - 2 * u.x.array / N_s
        PETSc.Sys.Print(f"Iteration: {it}, Error: {error:.2e}, Solve time: {t1 - t0:.3f}s")
        it += 1
    return error <= tol, it


def generate_plot_data(fun, L):
    pass


def i_p(y, phi, V, i0_prime, kappa, L_p):
    return (V_cell  * y /d_p - i_port(V=V0, i0_prime=i0_prime)/m(V=V0, i0_prime=i0_prime) - phi) / (R_p(L_p, kappa) * (1 - 1/(m(V=V0, i0_prime=i0_prime) * R_p(L_p, kappa))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='secondary current distribution')
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("-N_s", "--N_s", help="number of cells in stack", nargs='?', const=1, default=100, type=int)
    parser.add_argument("-L_p", "--L_p", help="Length [m]", nargs='?', const=1, default=0.02, type=float)
    parser.add_argument("-kappa", "--kappa", help="Conductivity [S/m]", nargs='?', const=1, default=4, type=float)
    parser.add_argument("-w", "--w", help="reaction penetration", nargs='?', const=1, default=100.0, type=float)
    parser.add_argument("-a", "--a", help="specific area [1/m]", nargs='?', const=1, default=100.0, type=float)
    parser.add_argument("-vary", "--vary", help="variable under study", nargs='?', const=1, default="L_p", type=str)

    args = parser.parse_args()
    workdir = args.mesh_folder
    results_dir = os.path.join(workdir, args.vary)
    utils.make_dir_if_missing(results_dir)
    
    a = args.a
    w = args.w
    i0 = w ** 2 * kappa * R * T / (faraday_const * a)
    N_s = args.N_s
    L = N_s * d_p/2
    L_p = args.L_p
    i0_prime = ((2 * a * i0 / alpha) * (kappa * R * T / faraday_const)) ** 0.5

    if args.vary == "L_p":
        results_path = os.path.join(results_dir, f"{L_p:.3f}.json")
    elif args.vary == "N_s":
        results_path = os.path.join(results_dir, f"{int(N_s)}.json")
    elif args.vary == 'w':
        results_path = os.path.join(results_dir, f"{w}.json")
    else:
        raise ValueError("Unknown study")

    comm = MPI.COMM_WORLD
    domain = mesh.create_interval(comm, 20000, [0, L])
    tdim = domain.topology.dim
    fdim = tdim - 1
    ft_imap = domain.topology.index_map(fdim)
    num_facets = ft_imap.size_local + ft_imap.num_ghosts
    indices = np.arange(0, num_facets)
    values = np.zeros(indices.shape, dtype=np.intc)
    left_marker = 1
    right_marker = 2

    values[0] = left_marker
    values[-1] = right_marker
    ft = mesh.meshtags(domain, fdim, indices, values)

    x = ufl.SpatialCoordinate(domain)
    n = ufl.FacetNormal(domain)
    V = fem.functionspace(domain, ("CG", 2))

    u, v = fem.Function(V), ufl.TestFunction(V)
    u0 = fem.Function(V)
    u0.interpolate(lambda x: x[0] - x[0] + V0)

    dx = ufl.Measure('dx', domain=domain)
    ds = ufl.Measure('ds', domain=domain, subdomain_data=ft)

    u_left = fem.Function(V)
    u_left.x.array[:] = 0
    domain.topology.create_connectivity(fdim, tdim)
    left_bc = fem.dirichletbc(
        u_left, fem.locate_dofs_topological(V, fdim, ft.find(left_marker))
    )

    bcs = [left_bc]
    tol = 1e-8
    max_its = 10
    run_model(u0=u0, u=u, v=v, dx=dx, bcs=bcs, N_s=N_s, kappa=kappa, L_p=L_p, i0_prime=i0_prime, tol=tol, max_its=max_its)

    bb_trees = bb_tree(domain, domain.topology.dim)
    n_points = 10000
    x = np.linspace(tol, L - tol, n_points)
    points = np.zeros((3, n_points))
    points[0] = x
    u_values = []
    cells = []
    points_on_proc = []
    cell_candidates = compute_collisions_points(bb_trees, points.T)
    colliding_cells = compute_colliding_cells(domain, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = u.eval(points_on_proc, cells)
    y = np.hstack((-points_on_proc[::-1, 0], points_on_proc[:, 0]))
    # x_lin = 0.5 * N_s + y/d_p
    u_bv = np.vstack((-u_values[::-1], u_values[:]))
    # y = np.hstack((-points_on_proc[::-1, 0], points_on_proc[:, 0]))
    i_p_bv = i_p(y, u_bv[:, 0], V=V0, i0_prime=i0_prime, kappa=kappa, L_p=L_p)
    W = fem.functionspace(domain, ("CG", 2))
    I_m = fem.Function(W)
    n = ufl.FacetNormal(domain)
    # t = ufl.as_vector((n[0]))
    I_m_expr = fem.Expression(kappa * A_m * grad(u), W.element.interpolation_points)
    I_m.interpolate(I_m_expr)
    lmda = lambda_squared(V=V0, kappa=kappa, L_p=L_p, i0_prime=i0_prime) ** 0.5
    results_json = {
        "I_ds [A]": I_ds(I_m, dx, L=L),
        # "I_m [A]":
        "i_p_max [A/m2]": np.max(i_p_bv),
        "lmda": lmda,
        "L": L,
        "I_m,max [A]": np.max(I_m.x.array),
    }
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False, indent=4)
