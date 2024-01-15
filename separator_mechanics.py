#!/usr/bin/env python3
import argparse
import json
import os
import timeit

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pyvista
import ufl
from basix.ufl import element
from collections import defaultdict
from dolfinx import cpp, default_scalar_type, fem, io, log, mesh, nls, plot
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import petsc
from dolfinx.io import VTXWriter
from mpi4py import MPI
from petsc4py import PETSc
from ufl import grad, inner

import commons, configs, constants

markers = commons.Markers()

MW_LI = 6.941e-3  # [kg.mol-1]
ρ_LI = 5.34e2  # [kg.m-3]
faraday_constant = 96485  # [C.mol-1]
L0 = 1e-6  # [m3]
E_LI = 5e9 # [Pa]  Lithium metal Elastic modulus


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coupling of stress and lithium metal/electrolyte active area fraction.')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="separator_mechanics")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1e-3)
    parser.add_argument("--Wa", help="Wagna number: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=np.inf)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='CONTACT_LOSS_SCALING', type=str)

    args = parser.parse_args()
    data_dir = os.path.join(f'{args.mesh_folder}')

    voltage = args.voltage
    comm = MPI.COMM_WORLD
    rank = comm.rank
    start_time = timeit.default_timer()

    scaling = configs.get_configs()[args.scaling]
    scale = [float(scaling[k]) for k in ['x', 'y', 'z']]
    dimensions = args.dimensions
    LX, LY, LZ = [float(v) * scale[idx] for (idx, v) in enumerate(dimensions.split("-"))]
    tetr_mesh_path = os.path.join(data_dir, 'tetr.xdmf')
    tria_mesh_path = os.path.join(data_dir, 'tria.xdmf')
    output_current_path = os.path.join(data_dir, 'current.bp')
    output_potential_path = os.path.join(data_dir, 'potential.bp')
    displacement_path = os.path.join(data_dir, 'displacement.bp')
    von_mises_path = os.path.join(data_dir, 'von_mises_stress.bp')
    frequency_path = os.path.join(data_dir, 'frequency.csv')
    simulation_metafile = os.path.join(data_dir, 'simulation.json')
    log.set_log_level(log.LogLevel.INFO)
    print("Loading tetrahedra (dim = 3) mesh..")
    with io.XDMFFile(comm, tetr_mesh_path, "r") as infile3:
        domain = infile3.read_mesh(cpp.mesh.GhostMode.none, 'Grid')
        ct = infile3.read_meshtags(domain, name="Grid")
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
    try:
        print("Attempting to load xmdf file for triangle mesh")
        with io.XDMFFile(comm, tria_mesh_path, "r") as infile2:
            ft = infile2.read_meshtags(domain, name="Grid")
        left_boundary = ft.find(markers.left)
        right_boundary = ft.find(markers.right)
    except RuntimeError as e:
        print("Missing xdmf file for triangle mesh!")
        facets = mesh.locate_entities_boundary(domain, dim=domain.topology.dim - 1,
                                               marker=lambda x: np.isfinite(x[2]))
        facets_l0 = mesh.locate_entities_boundary(domain, dim=domain.topology.dim - 1,
                                               marker=lambda x: np.isclose(x[2], 0))
        facets_lz = mesh.locate_entities_boundary(domain, dim=domain.topology.dim - 1,
                                               marker=lambda x: np.isclose(x[2], Lz))
        all_indices = set(tuple([val for val in facets]))
        l0_indices = set(tuple([val for val in facets_l0]))
        lz_indices = set(tuple([val for val in facets_lz]))
        insulator_indices = all_indices.difference(l0_indices | lz_indices)
        ft_indices = np.asarray(list(l0_indices) + list(lz_indices) + list(insulator_indices), dtype=np.int32)
        ft_values = np.asarray([markers.left_cc] * len(l0_indices) + [markers.right_cc] * len(lz_indices) + [markers.insulated] * len(insulator_indices), dtype=np.int32)
        left_boundary = facets_l0
        right_boundary = facets_lz
        ft = mesh.meshtags(domain, domain.topology.dim - 1, ft_indices, ft_values)

    # Dirichlet BCs
    V = fem.functionspace(domain, ("Lagrange", 2))
    u0 = fem.Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(voltage)

    u1 = fem.Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0.0)

    left_bc = fem.dirichletbc(u0, fem.locate_dofs_topological(V, 2, left_boundary))
    right_bc = fem.dirichletbc(u1, fem.locate_dofs_topological(V, 2, right_boundary))
    n = ufl.FacetNormal(domain)
    x = ufl.SpatialCoordinate(domain)
    metadata = {"quadrature_degree": 4}
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)

    # Define variational problem
    φ = ufl.TrialFunction(V)
    δφ = ufl.TestFunction(V)

    # bulk conductivity [S.m-1]
    kappa = fem.Constant(domain, PETSc.ScalarType(constants.KAPPA0))
    f = fem.Constant(domain, PETSc.ScalarType(0.0))
    g = fem.Constant(domain, PETSc.ScalarType(0.0))

    a = inner(kappa * grad(φ), grad(δφ)) * dx
    L = inner(f, δφ) * ufl.dx + inner(g, δφ) * ds(markers.insulated)

    options = {
               "ksp_type": "gmres",
               "pc_type": "hypre",
               "ksp_rtol": 1.0e-12
               }

    model = petsc.LinearProblem(a, L, bcs=[left_bc, right_bc], petsc_options=options)
    print('Solving potential problem..')
    uh = model.solve()

    with VTXWriter(comm, output_potential_path, [uh], engine="BP4") as vtx:
        vtx.write(0.0)

    W = fem.functionspace(domain, ("CG", 1, (3,)))
    current_expr = fem.Expression(-kappa * grad(uh), W.element.interpolation_points())
    current_h = fem.Function(W)
    current_h.interpolate(current_expr)

    with VTXWriter(comm, output_current_path, [current_h], engine="BP4") as vtx:
        vtx.write(0.0)

    I_left_cc = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(current_h, n) * ds(markers.left))), op=MPI.SUM)
    I_right_cc = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(current_h, n) * ds(markers.right))), op=MPI.SUM)
    print(f"Current at left = {I_left_cc:.2e} [A] and current at right = {I_right_cc:.2e}")

    # solution of stress
    print("Solving stress distribution problem")
    Q = fem.functionspace(domain, ("CG", 2, (3,)))

    # right boundary is assumed fixed
    u_bc = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
    right_dofs = fem.locate_dofs_topological(Q, ft.dim, ft.find(markers.right))
    bcs = [fem.dirichletbc(u_bc, right_dofs, Q)]

    # body force B and Piola traction vector P
    B = fem.Constant(domain, default_scalar_type((0, 0, 0)))
    # T = fem.Constant(domain, default_scalar_type((0, 0, 0)))

    # Piola-Kirchhoff stress at left boundary due to lithium growth velocity
    stress_expr = lambda t: fem.Expression(t * E_LI * (-kappa * grad(uh)) * MW_LI / (ρ_LI * faraday_constant * L0), Q.element.interpolation_points())
    T = fem.Function(Q)
    T.interpolate(stress_expr(1))

    u = fem.Function(Q)
    δu = ufl.TestFunction(Q)

    # Spatial dimension
    d = len(u)

    # Identity tensor
    I = ufl.variable(ufl.Identity(d))

    # Deformation gradient
    F = ufl.variable(I + ufl.grad(u))

    # Right Cauchy-Green tensor
    C = ufl.variable(F.T * F)

    # Invariants of deformation tensors
    Ic = ufl.variable(ufl.tr(C))
    J = ufl.variable(ufl.det(F))

    # Mechanical properties
    # Elasticity parameters
    E = default_scalar_type(1.0e4)
    ν = default_scalar_type(0.3)
    μ = fem.Constant(domain, E / (2 * (1 + ν)))
    λ = fem.Constant(domain, E * ν / ((1 + ν) * (1 - 2 * ν)))
    # Stored strain energy density (compressible neo-Hookean model)
    ψ = (μ / 2) * (Ic - 3) - μ * ufl.ln(J) + (λ / 2) * (ufl.ln(J))**2
    # Stress
    # Hyper-elasticity
    P = ufl.diff(ψ, F)
    # Define form F (we want to find u such that F(u) = 0)
    F_objective = inner(grad(δu), P) * dx - inner(δu, B) * dx - inner(δu, T) * ds(markers.left)
    problem = NonlinearProblem(F_objective, u, bcs)
    solver = NewtonSolver(domain.comm, problem)

    # Set Newton solver options
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.convergence_criterion = "incremental"
    num_its, converged = solver.solve(u)
    # write to file
    dvtx = VTXWriter(comm, displacement_path, [u], engine="BP4")
    dvtx.write(0.0)

    # visualization
    bb_trees = bb_tree(domain, domain.topology.dim)
    n_points = 10000
    tol = 1e-12
    x_viz = y_viz = np.zeros(n_points)
    z_viz = np.linspace(0 + tol, LZ - tol, n_points)

    points = np.zeros((3, n_points))
    points[0] = x_viz
    points[1] = y_viz
    points[2] = z_viz
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
    ax.plot(points_on_proc[:, 2], u_values, "k", linewidth=2)
    ax.grid(True)
    ax.set_xlim([0, LZ])
    # ax.set_ylim([0, C_MAX])
    ax.set_ylabel(r'$u$ [m]', rotation=0, labelpad=50, fontsize='xx-large')
    ax.set_xlabel('[m]')
    # ax.set_title(f't = {t:.1e} s, and D = {D_BULK:.1e} ' + r'm$^{2}$s$^{-1}$')
    plt.tight_layout()
    plt.savefig(os.path.join(args.mesh_folder, f'displacement.png'), dpi=1500)
    plt.close()

    # compute von mises stress
    # vmvtx = VTXWriter(comm, von_mises_path, [σ_vm], engine="BP4")
    # vmvtx.write(0.0)
    # vmvtx.close()

    # visualization
    pyvista.start_xvfb()
    plotter = pyvista.Plotter()
    plotter.open_gif(os.path.join(args.mesh_folder, "deformation.gif"), fps=3)

    topology, cells, geometry = plot.vtk_mesh(u.function_space)
    function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)

    values = np.zeros((geometry.shape[0], 3))
    values[:, :len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
    function_grid["u"] = values
    function_grid.set_active_vectors("u")

    # Warp mesh by deformation
    warped = function_grid.warp_by_vector("u", factor=1)
    warped.set_active_vectors("u")

    # Add mesh to plotter and visualize
    actor = plotter.add_mesh(warped, show_edges=True, lighting=False, clim=[0, 10])

    # Compute magnitude of displacement to visualize in GIF
    Vs = fem.FunctionSpace(domain, ("Lagrange", 2))
    magnitude = fem.Function(Vs)
    us = fem.Expression(ufl.sqrt(sum([u[i] ** 2 for i in range(len(u))])), Vs.element.interpolation_points())
    magnitude.interpolate(us)
    # warped["mag"] = magnitude.x.array
    for t in range(2, 10):
        T.interpolate(stress_expr(t))
        num_its, converged = solver.solve(u)
        dvtx.write(t)
        assert (converged)
        u.x.scatter_forward()
        print(f"Time step {n}, Number of iterations {num_its}")
        function_grid["u"][:, :len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
        magnitude.interpolate(us)
        # warped.set_active_scalars("mag")
        warped_n = function_grid.warp_by_vector(factor=1)
        plotter.update_coordinates(warped_n.points.copy(), render=False)
        plotter.update_scalar_bar_range([0, 10])
        plotter.update_scalars(magnitude.x.array)
        plotter.write_frame()
    plotter.close()
    dvtx.close()