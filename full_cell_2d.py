#!/usr/bin/env python3
import argparse
import os

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

from dolfinx import cpp, default_scalar_type, fem, io, mesh, nls, plot
from dolfinx.fem import petsc
from dolfinx.io import VTXWriter
from dolfinx.nls import petsc as petsc_nls
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 dot, div, dx, ds, dS, grad, inner, grad, avg, jump)

import commons, configs, geometry, utils

warnings.simplefilter('ignore')


def u_ocp_neg(soc):
    return 0.1


def u_ocp_pos(soc):
    if soc < 0 or soc > 1:
        raise ValueError("Invalid input value for state of discharge")

    # return (1 / 1.75) * (np.arctanh(-soc * 2.0 + 1) + 4.5)
    return 0.4


# def u_ocp_pos(sod, L=1, k=2):
#     return 2.5 + (1/k) * np.log((L - sod) / sod)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reaction Distribution')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="reaction_distribution")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    # parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    # parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1e-3)
    # parser.add_argument("--Wa", help="Wagna number: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=np.inf)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='MICRON_TO_METER', type=str)
    # parser.add_argument("--compute_distribution", help="compute current distribution stats", nargs='?', const=1, default=False, type=bool)
    # parser.add_argument("--dt", help="simulation timestep in seconds", nargs='?', const=1, default=1e-6, type=float)
    # parser.add_argument("--t_sim", help="integer number of simulation timesteps in dt", nargs='?', const=1, default=500, type=int)
    # parser.add_argument("--sod", help="state of discharge", nargs='?', const=1, default=0.975, type=float)

    args = parser.parse_args()

    markers = commons.Markers()
    soc = 0.5
    scaling = configs.get_configs()[args.scaling]
    scale = [float(scaling[val]) for val in ['x', 'y', 'z']]
    LX, LY, LZ = [int(val) * scale[idx] for (idx, val) in enumerate(args.dimensions.split("-"))]
    # mesh_folder = args.mesh_folder
    encoding = io.XDMFFile.Encoding.HDF5
    adaptive_refine = False
    micron = 1e-6
    name_of_study = args.name_of_study
    dimensions = args.dimensions
    resolution = 1
    markers = commons.Markers()
    workdir = f"output/{name_of_study}/{dimensions}/{resolution}"
    utils.make_dir_if_missing(workdir)
    output_meshfile = os.path.join(workdir, 'mesh.msh')
    tria_meshfile = os.path.join(workdir, "tria.xdmf")
    line_meshfile = os.path.join(workdir, "line.xdmf")
    potential_resultsfile = os.path.join(workdir, "potential.bp")
    potential_dg_resultsfile = os.path.join(workdir, "potential_dg.bp")
    concentration_resultsfile = os.path.join(workdir, "concentration.bp")
    current_resultsfile = os.path.join(workdir, "current.bp")

    comm = MPI.COMM_WORLD
    with io.XDMFFile(comm, tria_meshfile, "r") as infile3:
        domain = infile3.read_mesh(cpp.mesh.GhostMode.none, 'Grid')
        ct = infile3.read_meshtags(domain, name="Grid")
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)

    ft_imap = domain.topology.index_map(fdim)
    num_facets = ft_imap.size_local + ft_imap.num_ghosts
    indices = np.arange(0, num_facets)
    values = np.zeros(indices.shape, dtype=np.intc)  # all facets are tagged with zero

    with io.XDMFFile(comm, line_meshfile, "r") as infile2:
        ft = infile2.read_meshtags(domain, name="Grid")

    values[ft.indices] = ft.values
    meshtags = mesh.meshtags(domain, fdim, indices, values)
    domaintags = mesh.meshtags(domain, domain.topology.dim, ct.indices, ct.values)

    dx = ufl.Measure("dx", domain=domain, subdomain_data=domaintags)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=meshtags)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=meshtags)

    V = fem.FunctionSpace(domain, ("DG", 1))
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    n = ufl.FacetNormal(domain)
    x = ufl.SpatialCoordinate(domain)

    α = 10
    γ = 100

    h = ufl.CellDiameter(domain)
    h_avg = avg(h)

    x = SpatialCoordinate(domain)

    f = fem.Constant(domain, PETSc.ScalarType(0))
    g = fem.Constant(domain, PETSc.ScalarType(0))
    voltage = 1000e-3
    u_left = fem.Function(V)
    with u_left.vector.localForm() as u0_loc:
        u0_loc.set(0)
    u_right = fem.Function(V)
    with u_right.vector.localForm() as u1_loc:
        u1_loc.set(voltage)

    i0_neg = fem.Constant(domain, PETSc.ScalarType(1e2))
    i0_pos = fem.Constant(domain, PETSc.ScalarType(1e2))
    faraday_const = fem.Constant(domain, PETSc.ScalarType(96485))
    R = fem.Constant(domain, PETSc.ScalarType(8.3145))
    T = fem.Constant(domain, PETSc.ScalarType(298))
    U = ufl.as_vector((u_ocp_pos(soc), 0, 0))
    U_neg = ufl.as_vector((u_ocp_neg(soc), 0, 0))
    U_pos = ufl.as_vector((u_ocp_pos(soc), 0, 0))

    # κ = fem.Function(V)
    # Ω_neg_cc_cells = domaintags.find(markers.negative_cc)
    # Ω_neg_am_cells = domaintags.find(markers.negative_am)
    # Ω_se_cells = domaintags.find(markers.electrolyte)
    # Ω_pos_am_cells = domaintags.find(markers.positive_am)
    # Ω_pos_cc_cells = domaintags.find(markers.positive_cc)
    # κ.x.array[Ω_neg_cc_cells] = np.full_like(Ω_neg_cc_cells, 1, dtype=default_scalar_type)
    # κ.x.array[Ω_neg_am_cells] = np.full_like(Ω_neg_am_cells, 1, dtype=default_scalar_type)
    # κ.x.array[Ω_se_cells] = np.full_like(Ω_se_cells, 1, dtype=default_scalar_type)
    # κ.x.array[Ω_pos_am_cells] = np.full_like(Ω_pos_am_cells, 1, dtype=default_scalar_type)
    # κ.x.array[Ω_pos_cc_cells] = np.full_like(Ω_pos_cc_cells, 1, dtype=default_scalar_type)

    # formulation
    F = dot(grad(u), grad(v)) * dx - dot(v * n, grad(u)) * ds

    # Add DG/IP terms
    F += -dot(avg(grad(v)), jump(u, n)) * dS(0) - dot(jump(v, n), avg(grad(u))) * dS(0)
    F += (γ / h_avg) * dot(jump(v, n), jump(u, n)) * dS(0)
    F += α / h * v * u * ds(markers.left) + α / h * v * u * ds(markers.right)

    # negative am - electrolyte boundary
    F += - dot(avg(grad(v)), (R * T / i0_neg / faraday_const) * (grad(u))('-') + U_neg) * dS(markers.negative_am_v_electrolyte)
    F += + (α / h_avg) * dot(jump(v, n), (R * T / i0_neg / faraday_const) * (grad(u))('-') + U_neg) * dS(markers.negative_am_v_electrolyte)

    # electrolyte - positive am boundary
    F += - dot(avg(grad(v)), (R * T / i0_pos / faraday_const) * (grad(u))('-') + U_pos) * dS(markers.electrolyte_v_positive_am)
    F += + (α / h_avg) * dot(jump(v, n), (R * T / i0_pos / faraday_const) * (grad(u))('-') + U_pos) * dS(markers.electrolyte_v_positive_am)

    # Symmetry
    F += - dot(avg(grad(v)), jump(u, n)) * dS(markers.negative_am_v_electrolyte)
    F += - dot(avg(grad(v)), jump(u, n)) * dS(markers.electrolyte_v_positive_am)

    # Coercivity
    F += (α / h_avg) * dot(jump(v, n), jump(u, n)) * dS(markers.negative_am_v_electrolyte)
    F += (α / h_avg) * dot(jump(v, n), jump(u, n)) * dS(markers.electrolyte_v_positive_am)

    # Nitsche Dirichlet BC terms on left and right boundaries
    F += - dot(u * n, grad(v)) * ds(markers.left)
    F += u_left * dot(n, grad(v)) * ds(markers.left) - (α / h) * u_left * v * ds(markers.left)
    F += - dot(u * n, grad(v)) * ds(markers.right)
    F += u_right * dot(n, grad(v)) * ds(markers.right) - (α / h) * u_right * v * ds(markers.right)

    # Nitsche Neumann BC terms on insulated boundary
    F += -(h / α) * dot(g * n, grad(v)) * ds(markers.insulated)
    F += -g * v * ds(markers.insulated)

    # Source term
    F += -f * v * dx 

    problem = petsc.NonlinearProblem(F, u)
    solver = petsc_nls.NewtonSolver(comm, problem)
    solver.convergence_criterion = "residual"

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    # opts['maximum_iterations'] = 100
    ksp.setFromOptions()
    solver.solve(u)
    u.name = 'potential'

    # results to file
    V_CG1 = fem.FunctionSpace(domain, ("CG", 1))

    u_cg = fem.Function(V_CG1)
    u_cg.name = 'potential'
    u_cg.interpolate(u)

    with VTXWriter(comm, potential_dg_resultsfile, [u], engine="BP4") as vtx:
        vtx.write(0.0)
    with VTXWriter(comm, potential_resultsfile, [u_cg], engine="BP4") as vtx:
        vtx.write(0.0)

    W = fem.VectorFunctionSpace(domain, ("DG", 0))
    current_expr = fem.Expression(-grad(u), W.element.interpolation_points())
    current_h = fem.Function(W)
    current_h.name = 'current_density'
    current_h.interpolate(current_expr)

    V_CG0 = fem.VectorFunctionSpace(domain, ("CG", 1))
    current_cg = fem.Function(V_CG0)
    current_cg.name = 'current_density'
    current_cg_expr = fem.Expression(-grad(u_cg), V_CG0.element.interpolation_points())
    current_cg.interpolate(current_cg_expr)

    with VTXWriter(comm, current_resultsfile, [current_cg], engine="BP4") as vtx:
        vtx.write(0.0)

    I_left = fem.assemble_scalar(fem.form(inner(current_h, n) * ds(markers.left)))
    I_right = fem.assemble_scalar(fem.form(inner(current_h, n) * ds(markers.right)))
    print(f"out: {I_left:.2e} A, in: {abs(I_right):.2e} A")

    bb_trees = bb_tree(domain, domain.topology.dim)
    n_points = 10000
    tol = 1e-8  # Avoid hitting the outside of the domain
    x = np.linspace(tol, 165e-6 - tol, n_points)
    y = np.ones(n_points) * 0.5 * 825e-6  # midline
    points = np.zeros((3, n_points))
    points[0] = x
    points[1] = y
    u_values = []
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = compute_collisions_points(bb_trees, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = compute_colliding_cells(domain, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = u.eval(points_on_proc, cells)
    fig, ax = plt.subplots()
    ax.plot((1/micron) * points_on_proc[:, 0], u_values, "k", linewidth=2)
    # ax.grid(True)
    ax.set_xlim([0, 165])
    ax.set_ylabel(r'$\phi$ [V]', rotation=0, labelpad=30, fontsize='xx-large')
    ax.set_xlabel(r'[$\mu$m]')
    ax.set_title('Potential Across Midline')
    ax.axvline(x=20, linestyle='--', linewidth=0.5)
    ax.axvline(x=70, linestyle='--', linewidth=0.5)
    ax.axvline(x=95, linestyle='--', linewidth=0.5)
    ax.axvline(x=145, linestyle='--', linewidth=0.5)
    ax.axvline(x=165, linestyle='--', linewidth=0.5)
    ax.minorticks_on()
    ax.tick_params(which="both", left=True, right=True, bottom=True, top=True, labelleft=True, labelright=False, labelbottom=True, labeltop=False)
    plt.tight_layout()
    plt.show()

    # bb_trees = bb_tree(domain, domain.topology.dim)
    # points = np.zeros((3, n_points))
    # points[0] = x
    # points[1] = y
    # u_values = []
    # cells = []
    # points_on_proc = []
    # # Find cells whose bounding-box collide with the the points
    # cell_candidates = compute_collisions_points(bb_trees, points.T)
    # # Choose one of the cells that contains the point
    # colliding_cells = compute_colliding_cells(domain, cell_candidates, points.T)
    # for i, point in enumerate(points.T):
    #     if len(colliding_cells.links(i)) > 0:
    #         points_on_proc.append(point)
    #         cells.append(colliding_cells.links(i)[0])
    # points_on_proc = np.array(points_on_proc, dtype=np.float64)
    # current_values = current_h.eval(points_on_proc, cells)
    # fig, ax = plt.subplots()
    # ax.plot((1/micron) * points_on_proc[:, 0], np.linalg.norm(current_values, axis=1), "k", linewidth=2)
    # ax.grid(True)
    # # ax.set_xlim([50e-6, 200e-6])
    # # ax.set_ylim([0, 0.1])
    # ax.set_ylabel(r'$i$ [A/m$^2$]', rotation=0, labelpad=40, fontsize='xx-large')
    # ax.set_xlabel(r'[$\mu$m]')
    # ax.set_title('Current Density Across Midline')
    # plt.tight_layout()
    # plt.show()