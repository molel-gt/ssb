#!/usr/bin/env python3

import os
import timeit

import argparse
import dolfinx
import logging
import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import ufl

from dolfinx import cpp, fem, io, mesh, nls, plot
from mpi4py import MPI
from petsc4py import PETSc

import commons, configs, constants


markers = commons.SurfaceMarkers()
# model parameters
kappa = 1e3 # S/m
D = 1e-15  # m^2/s
F_c = 96485  # C/mol
i0 = 100  # A/m^2
dt = 1.0e-03
theta = 0.5  # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicholson
c_init = 0.01
R = 8.314
T = 298
# Lx = 20

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    # parser.add_argument('--grid_size', help='Lx-Ly-Lz', required=True)
    # parser.add_argument('--data_dir', help='Directory with tria.xdmf and tetr.xmf mesh files. Output files potential.xdmf and current.xdmf will be saved here', required=True, type=str)
    parser.add_argument("--voltage", help="Potential to set at the left current collector. Right current collector is set to a potential of 0", nargs='?', const=1, default=1)

    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    
    with io.XDMFFile(comm, "mesh/laminate/tria.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(cpp.mesh.GhostMode.shared_facet, name="Grid")
        ct = xdmf.read_meshtags(domain, name="Grid")

    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
    with io.XDMFFile(comm, "mesh/laminate/line.xdmf", "r") as xdmf:
        ft = xdmf.read_meshtags(domain, name="Grid")
    tags = mesh.meshtags(domain, domain.topology.dim - 1, ft.indices, ft.values)
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 2))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    f = dolfinx.fem.Constant(domain, PETSc.ScalarType(0.0))
    g = dolfinx.fem.Constant(domain, PETSc.ScalarType(0.0))
    kappa = dolfinx.fem.Constant(domain, PETSc.ScalarType(constants.KAPPA0))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=tags)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ds

    options = {
               "ksp_type": "gmres",
               "pc_type": "hypre",
               "ksp_rtol": 1.0e-12
               }

    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    ME = fem.FunctionSpace(domain, P1 * P1)
    x = ufl.SpatialCoordinate(domain)
    n = ufl.FacetNormal(domain)

    tags = mesh.meshtags(domain, domain.topology.dim - 1, ft.indices, ft.values)

    # Trial and test functions of the space `ME` are now defined:
    q, v = ufl.TestFunctions(ME)

    u = fem.Function(ME)  # current solution
    u0 = fem.Function(ME)  # solution from previous converged step

    # Split mixed functions
    c, mu = ufl.split(u)
    c0, mu0 = ufl.split(u0)

    # The initial conditions are interpolated into a finite element space
    def set_initial_bc(x):
        print(x.shape)
        new_x = x.T
        values = np.ones(new_x.shape[0])
        for i in range(x.shape[1]):
            val = c_init
            if np.isclose(x[0, i], 0):
                val = 0.0
            elif np.isclose(x[0, i], 19.9):
                if np.logical_and(np.less_equal(x[1, i], 17.5), np.greater_equal(x[1, i], 12.5)):
                    val = 0.0
                elif np.logical_and(np.less_equal(x[1, i], 7.5), np.greater_equal(x[1, i], 2.5)):
                    val = 0.0
            elif np.logical_and(np.less_equal(x[0, i], 19.9), np.isclose(x[1, i], [17.5, 12.5, 7.5, 2.5])).any():
                val = 0.0
            values[i] = val
        return values

    mu = ufl.variable(mu)
    flux = 0 * ( ufl.exp(0.5 * F_c * (mu - 0.0) / (R * T)) - ufl.exp(-0.5 * F_c * (mu - 0.0) / (R * T)))
    j_n = -D * ufl.dot(ufl.grad(c), n)
    flux = j_n
    flux2 = -kappa * ufl.dot(ufl.grad(mu), n)

    f = dolfinx.fem.Constant(domain, PETSc.ScalarType(0.0))
    g = dolfinx.fem.Constant(domain, PETSc.ScalarType(0.0))

    left_cc_facet = ft.find(markers.left_cc)
    right_cc_facet = ft.find(markers.right_cc)

    V0, dofs = ME.sub(0).collapse()
    V1, dofs = ME.sub(1).collapse()
    u_left = fem.Function(V1)
    u_right = fem.Function(V1)

    with u_left.vector.localForm() as u0_loc:
        u0_loc.set(0.0)

    with u_right.vector.localForm() as u0_loc:
        u0_loc.set(3.7)

    left_cc_dofs = dolfinx.fem.locate_dofs_topological(V1, 1, left_cc_facet)
    right_cc_dofs = dolfinx.fem.locate_dofs_topological(V1, 1, right_cc_facet)
    left_cc_bc = dolfinx.fem.dirichletbc(u_left, left_cc_dofs)
    right_cc_bc = dolfinx.fem.dirichletbc(u_right, right_cc_dofs)

    print(u.sub(0).x.array[:].shape)
    u.sub(0).interpolate(lambda x: set_initial_bc(x))
    u.x.scatter_forward()

    # mu_mid = (1.0 - theta) * mu0 + theta * mu
    # Weak statement of the equations
    c = ufl.variable(c)
    current = -D * ufl.diff(c, x)
    F0 = ufl.inner(c, q) * ufl.dx - ufl.inner(c0, q) * ufl.dx + dt * ufl.inner(D * ufl.grad(c), ufl.grad(q)) * ufl.dx - dt * ufl.inner(f, q) * ufl.dx - dt * ufl.inner(g, q) * ds(markers.insulated)
    F1 = ufl.inner(kappa * ufl.grad(mu), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx - ufl.inner(g, v) * ds(markers.insulated) - ufl.inner(flux, q) * ds(markers.left_cc)
    F = F0 + F1

    problem = fem.petsc.NonlinearProblem(F, u)
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.maximum_iterations = 250
    solver.rtol = 1e-12
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "gmres"
    opts[f"{option_prefix}pc_type"] = "hypre"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    # Output file
    file = io.XDMFFile(comm, "demo_ch/output.xdmf", "w")
    file.write_mesh(domain)

    # Step in time
    t = 0.0

    SIM_TIME = 12000 * dt

    # Get the sub-space for c and the corresponding dofs in the mixed space
    # vector
    V0, dofs = ME.sub(0).collapse()

    # Create a VTK 'mesh' with 'nodes' at the function dofs
    topology, cell_types, x = plot.create_vtk_mesh(V0)
    grid = pv.UnstructuredGrid(topology, cell_types, x)

    # Set output data
    grid.point_data["c"] = u.x.array[dofs].real
    grid.set_active_scalars("c")

    p = pvqt.BackgroundPlotter(title="concentration", auto_update=True)
    p.add_mesh(grid, clim=[0, c_init], cmap="hot")
    p.view_xy(True)
    p.add_text(f"time: {t}", font_size=12, name="timelabel")

    c = u.sub(0)
    u0.x.array[:] = u.x.array
    solution = fem.petsc.create_vector(problem.L)

    while (t < SIM_TIME):
        t += dt
        r = solver.solve(u)
        print(f"Step {int(t/dt)}: num iterations: {r[0]}")
        u0.x.array[:] = u.x.array
        file.write_function(c, t)

        # Update the plot window
        p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
        grid.point_data["c"] = u.x.array[dofs].real
        p.app.processEvents()

    file.close()

    # Update ghost entries and plot
    u.x.scatter_forward()
    grid.point_data["c"] = u.x.array[dofs].real