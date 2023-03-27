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
kappa = 1e-1 # S/m
D = 1e-10  # m^2/s
F_c = 96485  # C/mol
i0 = 100  # A/m^2
dt = 1.0e-03
t_iter = 200
theta = 0.5  # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicholson
c_init = 0.01
R = 8.314
T = 298


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
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

    f = dolfinx.fem.Constant(domain, PETSc.ScalarType(0.0))
    g = dolfinx.fem.Constant(domain, PETSc.ScalarType(0.0))
    kappa = dolfinx.fem.Constant(domain, PETSc.ScalarType(constants.KAPPA0))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=tags)

    # x = ufl.SpatialCoordinate(domain)
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
    q = ufl.TestFunction(V)

    u = fem.Function(V)  # current solution
    u0 = fem.Function(V)  # solution from previous converged step

    # The initial conditions are interpolated into a finite element space
    def set_initial_bc(x):
        new_x = x.T
        values = np.ones(new_x.shape[0])
        for i in range(x.shape[1]):
            values[i] = c_init
        return values

    u.interpolate(lambda x: set_initial_bc(x))
    u.x.scatter_forward()
    def get_solver(t):
        if t < 25:
            I = 1e-5
        else:
            I = 0
        f = dolfinx.fem.Constant(domain, PETSc.ScalarType(0.0))
        g1 = dolfinx.fem.Constant(domain, PETSc.ScalarType(I))
        g2 = dolfinx.fem.Constant(domain, PETSc.ScalarType(0.0))
        g3 = dolfinx.fem.Constant(domain, PETSc.ScalarType(0.0))
        F = ufl.inner(u, q) * ufl.dx 
        F += dt * ufl.inner(D * ufl.grad(u), ufl.grad(q)) * ufl.dx 
        F += dt * ufl.inner(f, q) * ufl.dx
        F += dt * ufl.inner(g1, q) * ds(markers.left_cc)
        F += dt * ufl.inner(g2, q) * ds(markers.right_cc)
        F += dt * ufl.inner(g3, q) * ds(markers.insulated)
        F -= ufl.inner(u0, q) * ufl.dx
    
        problem = fem.petsc.NonlinearProblem(F, u)
        solver = nls.petsc.NewtonSolver(comm, problem)
        solver.convergence_criterion = "incremental"
        solver.maximum_iterations = 50
        solver.rtol = 1e-14
        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "gmres"
        opts[f"{option_prefix}pc_type"] = "hypre"
        # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        ksp.setFromOptions()
        return solver
    
    W = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))
    # n = ufl.FacetNormal(domain)
    flux_expr = dolfinx.fem.Expression(D * ufl.sqrt(ufl.inner(ufl.grad(u), ufl.grad(u))), W.element.interpolation_points())
    flux_h = dolfinx.fem.Function(W)
    flux_h.interpolate(flux_expr)

    flux_fp = io.XDMFFile(comm, "flux.xdmf", "w")
    flux_fp.write_mesh(domain)
    flux_fp.write_function(flux_h, 0)
    # Output file
    file = io.XDMFFile(comm, "concentration.xdmf", "w")
    file.write_mesh(domain)
    file.write_function(u, 0)

    # Step in time
    t = 0.0

    SIM_TIME = t_iter * dt

    # Create a VTK 'mesh' with 'nodes' at the function dofs
    topology, cell_types, x = plot.create_vtk_mesh(V)
    grid = pv.UnstructuredGrid(topology, cell_types, x)

    # Set output data
    grid.point_data["c"] = u.x.array
    grid.set_active_scalars("c")

    p = pvqt.BackgroundPlotter(title="concentration", auto_update=True)
    p.add_mesh(grid, clim=[0, c_init], cmap="hot", name='mesh')
    p.view_xy(True)
    p.add_text(f"time: {t}", font_size=12, name="timelabel")

    u0.x.array[:] = u.x.array

    while (t < SIM_TIME):
        t += dt
        rsolver = get_solver(t)
        r = rsolver.solve(u)
        print(f"Step {int(t/dt)}: num iterations: {r[0]}")
        u0.x.array[:] = u.x.array
        if np.any(np.isclose(u0.x.array[:],  0)):
            break
        file.write_function(u, t)
        tot_c = dolfinx.fem.assemble_scalar(dolfinx.fem.form(u * ufl.dx)) 
        vol = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain)))
        avg_c = tot_c / vol
        print("δξ:", (c_init - avg_c) / c_init)
        flux_h.interpolate(flux_expr)
        flux_fp.write_function(flux_h, t)

        # Update the plot window
        p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
        grid.point_data["c"] = u.x.array
        p.app.processEvents()

    file.close()
    flux_fp.close()

    # Update ghost entries and plot
    u.x.scatter_forward()
    grid.point_data["c"] = u.x.array