#!/usr/bin/env python3
import argparse
import json
import os
import timeit

import dolfinx
import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
import ufl
import warnings

from basix.ufl import element
from dolfinx import cpp, default_scalar_type, fem, graph, io, mesh, nls, plot
from dolfinx.fem import petsc
from dolfinx.graph import partitioner_parmetis
from dolfinx.io import gmshio, VTXWriter
from dolfinx.nls import petsc as petsc_nls
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 dot, div, dx, ds, dS, grad, inner, grad, avg, jump)

import commons, configs, geometry, utils

warnings.simplefilter('ignore')

kappa_se = 0.1
kappa_am = 0.2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="reaction_distribution")
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--dimensions", help="integer representation of LX-LY-LZ of the grid", required=True)
    parser.add_argument('--lsep', help=f'integer representation of separator thickness', type=int, required=True)
    parser.add_argument('--radius', help=f'integer representation of +AM particle radius', type=int, required=True)
    parser.add_argument('--eps_am', help=f'positive active material volume fraction', type=float, required=True)
    parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1, type=float)
    parser.add_argument("--Wa", help="Wagna number: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, type=float, default=1e-5)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='CONTACT_LOSS_SCALING', type=str)
    parser.add_argument("--compute_distribution", help="compute current distribution stats", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    start_time = timeit.default_timer()
    markers = commons.Markers()
    LX, LY, LZ = [int(v) for v in args.dimensions.split("-")]
    scaling = configs.get_configs()[args.scaling]
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])
    LX = LX * scale_x
    LY = LY * scale_y
    LZ = LZ * scale_z
    Rp = args.radius * scale_x
    Lsep = args.lsep * scale_x
    Lcat = LZ - Lsep

    comm = MPI.COMM_WORLD
    msh_path = os.path.join(args.mesh_folder, "mesh.msh")
    workdir = os.path.join(args.mesh_folder, str(args.Wa))
    potential_resultsfile = os.path.join(workdir, "potential.bp")
    potential_dg_resultsfile = os.path.join(workdir, "potential_dg.bp")
    concentration_resultsfile = os.path.join(workdir, "concentration.bp")
    current_resultsfile = os.path.join(workdir, "current.bp")
    current_dg_resultsfile = os.path.join(workdir, "current_dg.bp")
    simulation_metafile = os.path.join(workdir, 'simulation.json')
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = gmshio.read_from_msh(msh_path, comm, partitioner=partitioner)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)
    ft_imap = domain.topology.index_map(fdim)
    num_facets = ft_imap.size_local + ft_imap.num_ghosts
    indices = np.arange(0, num_facets)
    values = np.zeros(indices.shape, dtype=np.intc)  # all facets are tagged with zero
    values[ft.indices] = ft.values
    ft = mesh.meshtags(domain, fdim, indices, values)
    ct = mesh.meshtags(domain, domain.topology.dim, ct.indices, ct.values)
    left_boundary = ft.find(markers.left)
    right_boundary = ft.find(markers.right)

    print("Finished loading mesh.")

    dx = ufl.Measure("dx", domain=domain, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=ft)

    V = fem.FunctionSpace(domain, ("DG", 1))
    W = fem.functionspace(domain, ("CG", 1, (3,)))
    Q = fem.FunctionSpace(domain, ("DG", 0))
    u = fem.Function(V, name='potential')
    v = ufl.TestFunction(V)
    n = ufl.FacetNormal(domain)
    x = ufl.SpatialCoordinate(domain)

    h = ufl.CellDiameter(domain)
    h_avg = avg(h)

    f = fem.Constant(domain, PETSc.ScalarType(0))
    g = fem.Constant(domain, PETSc.ScalarType(0))

    u_left = fem.Function(V)
    with u_left.vector.localForm() as u0_loc:
        u0_loc.set(0)
    u_right = fem.Function(V)
    with u_right.vector.localForm() as u1_loc:
        u1_loc.set(args.voltage)

    
    faraday_const = fem.Constant(domain, PETSc.ScalarType(96485))
    R = fem.Constant(domain, PETSc.ScalarType(8.3145))
    T = fem.Constant(domain, PETSc.ScalarType(298))
    i0_left = kappa_se * R * T / (LZ * args.Wa * faraday_constant)
    i0_interface = fem.Constant(domain, PETSc.ScalarType(1e2))
    def ocv(sod, L=1, k=2):
        return 2.5 + (1/k) * np.log((L - sod) / sod)

    U = ufl.as_vector((ocv(0.975), ocv(0.975), ocv(0.975)))
    U = ufl.as_vector((0.15, 0.15, 15))

    kappa = fem.Function(Q)
    cells_electrolyte = ct.find(markers.electrolyte)
    cells_pos_am = ct.find(markers.positive_am)
    kappa.x.array[cells_electrolyte] = np.full_like(cells_electrolyte, kappa_se, dtype=default_scalar_type)
    kappa.x.array[cells_pos_am] = np.full_like(cells_pos_am, kappa_am, dtype=default_scalar_type)

    alpha = 10
    gamma = 10

    F = kappa * inner(grad(u), grad(v)) * dx - f * v * dx - kappa * inner(grad(u), n) * v * ds

    # Add DG/IP terms
    F += - avg(kappa) * inner(jump(u, n), avg(grad(v))) * dS(0)
    F += - inner(jump(kappa * u, n), avg(grad(v))) * dS(0)
    F += - inner(avg(kappa * grad(u)), jump(v, n)) * dS(0)
    F += + avg(u) * inner(jump(kappa, n), avg(grad(v))) * dS(0)
    F += alpha / h_avg * inner(jump(v, n), jump(u, n)) * dS(0)

    # Internal boundary
    F += - avg(kappa) * dot(avg(grad(v)), (R * T / i0_interface / faraday_const) * (kappa * grad(u))('+') + U) * dS(markers.electrolyte_v_positive_am)
    F += (alpha / h_avg) * avg(kappa) * dot(jump(v, n), (R * T / i0_interface / faraday_const) * (kappa * grad(u))('+') + U) * dS(markers.electrolyte_v_positive_am)

    # # Symmetry
    F += - avg(kappa) * inner(jump(u, n), avg(grad(v))) * dS(markers.electrolyte_v_positive_am)

    # # Coercivity
    F += alpha / h_avg * avg(kappa) * inner(jump(u, n), jump(v, n)) * dS(markers.electrolyte_v_positive_am)

    # Nitsche Dirichlet BC terms on left and right boundaries
    F += - kappa * (u - u_left) * inner(n, grad(v)) * ds(markers.left)
    F += gamma / h * (u - u_left) * v * ds(markers.left)
    F += - kappa * (u - u_right) * inner(n, grad(v)) * ds(markers.right)
    F += gamma / h * (u - u_right) * v * ds(markers.right)

    # Nitsche Neumann BC terms on insulated boundary
    F += -g * v * ds(markers.insulated) + gamma * h * g * inner(grad(v), n) * ds(markers.insulated_electrolyte)
    F += - gamma * h * inner(inner(grad(u), n), inner(grad(v), n)) * ds(markers.insulated_electrolyte)
    F += -g * v * ds(markers.insulated) + 10 * gamma * h * g * inner(grad(v), n) * ds(markers.insulated_positive_am)
    F += - 10 * gamma * h * inner(inner(grad(u), n), inner(grad(v), n)) * ds(markers.insulated_positive_am)

    problem = petsc.NonlinearProblem(F, u)
    solver = petsc_nls.NewtonSolver(comm, problem)
    solver.convergence_criterion = "residual"
    solver.maximum_iterations = 25
    solver.atol = 1e-12
    solver.rtol = 1e-11
    # solver.atol = np.finfo(float).eps
    # solver.rtol = np.finfo(float).eps * 10

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "gmres"
    opts[f"{option_prefix}pc_type"] = "hypre"
    ksp.setFromOptions()
    n_iters, converged = solver.solve(u)
    if not converged:
        print(f"Not converged in {n_iters} iterations")
    else:
        print(f"Converged in {n_iters} iterations")

    current_expr = fem.Expression(-kappa * grad(u), W.element.interpolation_points())
    current_h = fem.Function(W, name='current_density')
    current_h.interpolate(current_expr)

    with VTXWriter(comm, current_dg_resultsfile, [current_h], engine="BP4") as vtx:
        vtx.write(0.0)

    V_CG0 = fem.VectorFunctionSpace(domain, ("CG", 1))
    current_cg = fem.Function(V_CG0)
    current_cg.name = 'current_density'
    current_cg_expr = fem.Expression(-kappa * grad(u_cg), V_CG0.element.interpolation_points())
    current_cg.interpolate(current_cg_expr)

    with VTXWriter(comm, current_resultsfile, [current_cg], engine="BP4") as vtx:
        vtx.write(0.0)

    I_left = fem.assemble_scalar(fem.form(inner(current_h, n) * ds(markers.left)))
    I_interface = fem.assemble_scalar(fem.form(inner(current_h, n) * ds(markers.electrolyte_v_positive_am)))
    I_right = fem.assemble_scalar(fem.form(inner(current_h, n) * ds(markers.right)))
    I_insulated_sse = fem.assemble_scalar(fem.form(np.abs(inner(current_h, n)) * ds(markers.insulated_electrolyte)))
    I_insulated_pos_am = fem.assemble_scalar(fem.form(np.abs(inner(current_h, n)) * ds(markers.insulated_positive_am)))
    print(f"left: {I_left:.2e} A, right: {abs(I_right):.2e} A")
    print(f"insulated electrolyte: {I_insulated_sse:.2e} A, insulated positive am: {I_insulated_pos_am:.2e} A")
    print(f"Float precision: {np.finfo(float).eps}")

    # C = fem.FunctionSpace(domain, ("CG", 1))  # concentration
