#!/usr/bin/env python3
import argparse
import json
import os

import dolfinx
import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
import ufl
import warnings

from dolfinx import cpp, default_scalar_type, fem, io, mesh, nls, plot
from dolfinx.fem import petsc
from dolfinx.io import VTXWriter
from dolfinx.nls import petsc as petsc_nls
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (CellDiameter, Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 dot, div, dx, ds, dS, grad, inner, grad, avg, jump)

import commons, configs, geometry, utils

warnings.simplefilter('ignore')

C_INIT = 1000  # [mol/m3]
C_MAX = C_INIT
D_BULK = 1e1  # [m2.s-1]
# https://periodictable.com/Properties/A/ElectricalConductivity.an.html
SIGMA_COPPER = 5.8e7  # [S.m-1]
SIGMA_LITHIUM = 1.1e7  # [S.m-1]
# Chen 2020
KAPPA_ELECTROLYTE = 2.5e1  # [S.m-1]

encoding = io.XDMFFile.Encoding.HDF5


def open_circuit_voltage(sod, L=1, k=2):
    return 2.5 + (1/k) * np.log((L - sod) / sod)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Current Distribution')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="lmb_3d_cc")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1e-3)
    parser.add_argument("--Wa", help="Wagna number: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=np.inf)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='MICRON_TO_METER', type=str)
    parser.add_argument("--compute_distribution", help="compute current distribution stats", nargs='?', const=1, default=False, type=bool)
    parser.add_argument("--dt", help="simulation timestep in seconds", nargs='?', const=1, default=1e-6, type=float)
    parser.add_argument("--t_sim", help="integer number of simulation timesteps in dt", nargs='?', const=1, default=500, type=int)
    parser.add_argument("--sod", help="state of discharge", nargs='?', const=1, default=0.975, type=float)

    args = parser.parse_args()

    markers = commons.Markers()
    comm = MPI.COMM_WORLD
    scaling = configs.get_configs()[args.scaling]
    scale = [float(scaling[val]) for val in ['x', 'y', 'z']]
    LX, LY, LZ = [int(val) * scale[idx] for (idx, val) in enumerate(args.dimensions.split("-"))]
    mesh_folder = args.mesh_folder
    results_folder = os.path.join(mesh_folder, str(args.Wa), f"{args.dt:.0e}")
    utils.make_dir_if_missing(results_folder)
    output_meshfile = os.path.join(mesh_folder, 'mesh.msh')
    tetr_meshfile = os.path.join(mesh_folder, "tetr.xdmf")
    tria_meshfile = os.path.join(mesh_folder, "tria.xdmf")
    potential_resultsfile = os.path.join(results_folder, "potential.bp")
    potential_dg_resultsfile = os.path.join(results_folder, "potential_dg.bp")
    concentration_resultsfile = os.path.join(results_folder, "concentration.bp")
    current_resultsfile = os.path.join(results_folder, "current.bp")
    simulation_meta_file = os.path.join(results_folder, "simulation.json")

    # line plots
    n_points = 10000

    # Load mesh files
    with io.XDMFFile(comm, tetr_meshfile, "r") as infile3:
        domain = infile3.read_mesh(cpp.mesh.GhostMode.none, 'Grid')
        ct = infile3.read_meshtags(domain, name="Grid")
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)
    
    ft_imap = domain.topology.index_map(fdim)
    num_facets = ft_imap.size_local + ft_imap.num_ghosts
    indices = np.arange(0, num_facets)
    values = np.zeros(indices.shape, dtype=np.intc)  # all facets are tagged with zero
    
    with io.XDMFFile(comm, tria_meshfile, "r") as infile2:
        ft = infile2.read_meshtags(domain, name="Grid")

    # domain and subdomain markers
    values[ft.indices] = ft.values
    meshtags = mesh.meshtags(domain, fdim, indices, values)
    domaintags = mesh.meshtags(domain, domain.topology.dim, ct.indices, ct.values)

    dx = ufl.Measure("dx", domain=domain, subdomain_data=domaintags)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=meshtags)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=meshtags)

    n = FacetNormal(domain)
    x = SpatialCoordinate(domain)

    # penalty parameters
    alpha = 10
    gamma = 100

    h = CellDiameter(domain)
    h_avg = avg(h)

    # trial/test function spaces
    V = fem.FunctionSpace(domain, ("DG", 1))
    u = fem.Function(V)
    δu = ufl.TestFunction(V)

    # subdomain conductivity values
    kappa = fem.Function(V)
    cells_negative_cc = ct.find(markers.negative_cc)
    cells_electrolyte = ct.find(markers.electrolyte)

    kappa.x.array[cells_negative_cc] = np.full_like(cells_negative_cc, SIGMA_COPPER, dtype=default_scalar_type)
    kappa.x.array[cells_electrolyte] = np.full_like(cells_electrolyte, KAPPA_ELECTROLYTE, dtype=default_scalar_type)

    f = fem.Constant(domain, PETSc.ScalarType(0))
    g = fem.Constant(domain, PETSc.ScalarType(0))

    # parameters
    i0 = fem.Constant(domain, PETSc.ScalarType(1e2))
    faraday_const = fem.Constant(domain, PETSc.ScalarType(96485))
    R = fem.Constant(domain, PETSc.ScalarType(8.3145))
    T = fem.Constant(domain, PETSc.ScalarType(298))

    # dirichlet boundary conditions for potentiostatic
    u_left = fem.Function(V)
    with u_left.vector.localForm() as u0_loc:
        u0_loc.set(0)
    u_right = fem.Function(V)
    with u_right.vector.localForm() as u1_loc:
        u1_loc.set(args.voltage)

    U_ocv = ufl.as_vector((open_circuit_voltage(args.sod), 0, 0))

    F = dot(grad(u), grad(δu)) * dx - dot(δu * n, grad(u)) * ds

    # Add DG/IP terms
    F += - dot(avg(grad(δu)), jump(u, n)) * dS(0) - dot(jump(δu, n), avg(grad(u))) * dS(0)
    F += (gamma / h_avg) * dot(jump(δu, n), jump(u, n)) * dS(0)
    F += alpha / h * δu * u * ds(markers.left) + alpha / h * δu * u * ds(markers.right)

    # Internal boundary
    F += - dot(avg(grad(δu)), (R * T / i0 / faraday_const) * grad(u)('-') + U_ocv) * dS(markers.middle)
    F += (alpha / h_avg) * dot(jump(δu, n), (R * T / i0 / faraday_const) * grad(u)('-') + U_ocv) * dS(markers.middle)

    # Symmetry
    F += - dot(avg(grad(δu)), jump(u, n)) * dS(markers.middle)

    # Coercivity
    F += alpha / h_avg * dot(jump(δu, n), jump(u, n)) * dS(markers.middle)

    # Nitsche Dirichlet BC terms on left and right boundaries
    F += - dot(u * n, grad(δu)) * ds(markers.left)
    F += u_left * dot(n, grad(δu)) * ds(markers.left) - (alpha / h) * u_left * δu * ds(markers.left)
    F += - dot(u * n, grad(δu)) * ds(markers.right)
    F += u_right * dot(n, grad(δu)) * ds(markers.right) - (alpha / h) * u_right * δu * ds(markers.right)

    # Nitsche Neumann BC terms on insulated boundary
    F += -(h / alpha) * dot(g * n, grad(δu)) * ds(markers.insulated_negative_cc)
    F += -g * δu * ds(markers.insulated_negative_cc)
    F += -(h / alpha) * dot(g * n, grad(δu)) * ds(markers.insulated_electrolyte)
    F += -g * δu * ds(markers.insulated_electrolyte)

    # Source term
    F += -f * δu * dx

    problem = petsc.NonlinearProblem(F, u)
    solver = petsc_nls.NewtonSolver(comm, problem)
    solver.convergence_criterion = "residual"
    solver.maximum_iterations = 100

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "gmres"
    opts[f"{option_prefix}pc_type"] = "hypre"
    ksp.setFromOptions()
    solver.solve(u)
    u.name = 'potential'

    V_CG1 = fem.FunctionSpace(domain, ("CG", 1))
    u_cg = fem.Function(V_CG1)
    u_cg.name = 'potential'
    u_cg.interpolate(u)

    V_CG1 = fem.FunctionSpace(domain, ("CG", 1))

    u_cg = fem.Function(V_CG1)
    u_cg.name = 'potential'
    u_cg.interpolate(u)

    with VTXWriter(comm, potential_dg_resultsfile, [u], engine="BP4") as vtx:
        vtx.write(0.0)
    with VTXWriter(comm, potential_resultsfile, [u_cg], engine="BP4") as vtx:
        vtx.write(0.0)

    I_sup = fem.assemble_scalar(fem.form(inner(grad(u_cg), n) * ds(markers.left)))
    area_at_left = fem.assemble_scalar(fem.form(1 * ds(markers.left)))
    print(f"Current at terminal of negative current collector is: {I_sup} A")

    bb_trees = bb_tree(domain, domain.topology.dim)
    tol = 1e-8  # Avoid hitting the outside of the domain

    # midline in 3D
    x_viz = y_viz = np.zeros(n_points)
    z_viz = np.linspace(0 + tol, LZ - tol, n_points)# midline

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
    ax.set_ylim([0, args.voltage])
    ax.set_ylabel(r'$\phi$ [V]', rotation=0, labelpad=30, fontsize='xx-large')
    ax.set_xlabel('[m]')
    ax.set_title('Potential Across Midline')
    plt.tight_layout()
    plt.savefig(os.path.join(mesh_folder, "potential-midline.png"), dpi=1500)

    ##### Concentration Problem
    full_mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(output_meshfile, comm, 0)

    # Create submesh for pe
    positive_am_domain, entity_map, vertex_map, geom_map = dolfinx.mesh.create_submesh(full_mesh, full_mesh.topology.dim, cell_tags.indices[(cell_tags.values == markers.electrolyte)])

    # Transfer facet tags from parent mesh to submesh
    tdim = full_mesh.topology.dim
    fdim = tdim - 1
    c_to_f = full_mesh.topology.connectivity(tdim, fdim)
    f_map = full_mesh.topology.index_map(fdim)
    all_facets = f_map.size_local + f_map.num_ghosts
    all_values = np.zeros(all_facets, dtype=np.int32)
    all_values[facet_tags.indices] = facet_tags.values

    positive_am_domain.topology.create_entities(fdim)
    subf_map = positive_am_domain.topology.index_map(fdim)
    positive_am_domain.topology.create_connectivity(tdim, fdim)
    c_to_f_sub = positive_am_domain.topology.connectivity(tdim, fdim)
    num_sub_facets = subf_map.size_local + subf_map.num_ghosts
    sub_values = np.empty(num_sub_facets, dtype=np.int32)
    for i, entity in enumerate(entity_map):
        parent_facets = c_to_f.links(entity)
        child_facets = c_to_f_sub.links(i)
        for child, parent in zip(child_facets, parent_facets):
            sub_values[child] = all_values[parent]
    sub_meshtag = dolfinx.mesh.meshtags(positive_am_domain, positive_am_domain.topology.dim - 1, np.arange(
        num_sub_facets, dtype=np.int32), sub_values)
    positive_am_domain.topology.create_connectivity(positive_am_domain.topology.dim - 1, positive_am_domain.topology.dim)

    with dolfinx.io.XDMFFile(comm, os.path.join(mesh_folder, "submesh.xdmf"), "w", encoding=encoding) as xdmf:
        xdmf.write_mesh(positive_am_domain)
        xdmf.write_meshtags(sub_meshtag, x=positive_am_domain.geometry)

    eps = 1e-15
    dt = args.dt
    t = 0 # Start time
    T = args.t_sim * args.dt

    dx = ufl.Measure("dx", domain=positive_am_domain)
    ds = ufl.Measure("ds", domain=positive_am_domain, subdomain_data=sub_meshtag)
    dS = ufl.Measure("dS", domain=positive_am_domain, subdomain_data=sub_meshtag)
    n = ufl.FacetNormal(positive_am_domain)
    tdim = positive_am_domain.topology.dim
    fdim = tdim - 1

    # Create boundary condition
    # boundary_facets = sub_meshtag.find(markers.middle)
    # bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(Q, fdim, boundary_facets), Q)

    Q = fem.FunctionSpace(positive_am_domain, ("CG", 1))
    c_n = fem.Function(Q)
    c_n.name = "c_n"
    c_n.interpolate(lambda x:  x[0] - x[0] + C_INIT)
    c_n.x.scatter_forward()

    potential = fem.Function(Q)
    padding = 1e-16
    u1_2_u2_nmm_data = \
            fem.create_nonmatching_meshes_interpolation_data(
                potential.function_space.mesh._cpp_object,
                potential.function_space.element,
                u_cg.function_space.mesh._cpp_object, padding=padding)

    potential.interpolate(u, nmm_interpolation_data=u1_2_u2_nmm_data)
    potential.x.scatter_forward()

    ch = fem.Function(Q)
    ch.name = "concentration"
    ch.interpolate(lambda x: x[0] - x[0] + C_INIT)
    ch.x.scatter_forward()

    c = ufl.TrialFunction(Q)
    δc = ufl.TestFunction(Q)

    f = fem.Constant(positive_am_domain, PETSc.ScalarType(0))
    g = fem.Constant(positive_am_domain, PETSc.ScalarType(0))
    g_middle = fem.Constant(positive_am_domain, PETSc.ScalarType(0))
    D = fem.Constant(positive_am_domain, PETSc.ScalarType(D_BULK))

    a = c * δc * dx + dt * inner(D * grad(c), grad(δc)) * dx
    L = (
        (c_n + dt * f) * δc * dx 
        + dt * inner(g, δc) * ds(markers.insulated_electrolyte) 
        + dt * inner(g, δc) * ds(markers.right)
        + dt * inner(grad(potential) / 96485, n) * δc * ds(markers.middle)
    )

    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    A = petsc.assemble_matrix(bilinear_form, bcs=[])
    A.assemble()
    b = fem.petsc.create_vector(linear_form)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)

    c_vtx = VTXWriter(comm, concentration_resultsfile, [ch], engine="BP4")
    c_vtx.write(0.0)

    while t < T:
        t += dt

        A = fem.petsc.assemble_matrix(bilinear_form, bcs=[])
        A.assemble()
        solver.setOperators(A)

        # Update the right hand side reusing the initial vector
        with b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(b, linear_form)

        # Apply Dirichlet boundary condition to the vector
        fem.petsc.apply_lifting(b, [bilinear_form], [[]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, [])

        # Solve linear problem
        solver.solve(b, ch.vector)
        ch.x.scatter_forward()
        # c_avg = fem.assemble_scalar(fem.form(ch * dx)) / fem.assemble_scalar(fem.form(1 * dx))
        # print(f"average concentration: {c_avg}")

        # Update solution at previous time step (c_n)
        if np.any(ch.x.array < 0):
            print(f"Lithium depletion at {t:.2e} seconds")
            break
        if np.any(ch.x.array == np.inf):
            print(f"diverged at {t:.2e} seconds")
            break
        c_n.x.array[:] = ch.x.array
        c_vtx.write(t)
    c_vtx.close()

    # visualization
    bb_trees = bb_tree(positive_am_domain, positive_am_domain.topology.dim)
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
    colliding_cells = compute_colliding_cells(positive_am_domain, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = ch.eval(points_on_proc, cells)
    fig, ax = plt.subplots()
    ax.plot(points_on_proc[:, 2], u_values, "k", linewidth=2)
    ax.grid(True)
    ax.set_xlim([0, LZ])
    ax.set_ylim([0, C_MAX])
    ax.set_ylabel(r'$c$ [mol/m$^3$]', rotation=0, labelpad=50, fontsize='xx-large')
    ax.set_xlabel('[m]')
    ax.set_title(f't = {t:.1e} s, and D = {D_BULK:.1e} ' + r'm$^{2}$s$^{-1}$')
    plt.tight_layout()
    plt.savefig(os.path.join(mesh_folder, f'concentration-midline-{D_BULK:.2e}.png'), dpi=1500)
    plt.close()
