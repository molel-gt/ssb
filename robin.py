#!/usr/bin/env python3

import dolfinx
import numpy as np
import ufl

from dolfinx import mesh, fem, io, nls, log

from mpi4py import MPI
from petsc4py import PETSc
from ufl import (dx, grad, inner, lhs, rhs)


i0 = 10  # exhchange current density
F_c = 96485  # Faraday constant
R = 8.314
T = 298
alpha_a = 0.5
alpha_c = 0.5

mesh2d = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

x = ufl.SpatialCoordinate(mesh2d)

# Define physical parameters and boundary condtions
f = dolfinx.fem.Constant(mesh2d, PETSc.ScalarType(0.0))
n = ufl.FacetNormal(mesh2d)
g = dolfinx.fem.Constant(mesh2d, PETSc.ScalarType(0.0))
g1 = dolfinx.fem.Constant(mesh2d, PETSc.ScalarType(1e-3))
kappa = fem.Constant(mesh2d, PETSc.ScalarType(1))

# Define function space and standard part of variational form
V = fem.FunctionSpace(mesh2d, ("CG", 1))
u, v = fem.Function(V), ufl.TestFunction(V)
F = kappa * inner(grad(u), grad(v)) * dx - inner(f, v) * dx
g_curr = i0 * ufl.exp(u/R/T)
boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], 1)),
              (3, lambda x: np.isclose(x[1], 0)),
              (4, lambda x: np.isclose(x[1], 1))]

facet_indices, facet_markers = [], []
fdim = mesh2d.topology.dim - 1
for (marker, locator) in boundaries:
    facets = mesh.locate_entities(mesh2d, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(mesh2d, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

mesh2d.topology.create_connectivity(mesh2d.topology.dim - 1, mesh2d.topology.dim)
with io.XDMFFile(mesh2d.comm, "facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh2d)
    xdmf.write_meshtags(facet_tag)

ds = ufl.Measure("ds", domain=mesh2d, subdomain_data=facet_tag)
u0 = dolfinx.fem.Function(V)
with u0.vector.localForm() as u0_loc:
    u0_loc.set(0)

x0facet = dolfinx.mesh.locate_entities_boundary(mesh2d, 1, lambda x: np.isclose(x[0], 0.0))
x0bc = dolfinx.fem.dirichletbc(u0, dolfinx.fem.locate_dofs_topological(V, 1, x0facet))

F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx - ufl.inner(g1, v) * ds(2) - ufl.inner(g, v) * ds(3) - ufl.inner(g, v) * ds(4)

problem = fem.petsc.NonlinearProblem(F, u, bcs=[x0bc])
solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.maximum_iterations = 100
solver.report = True

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

log.set_log_level(log.LogLevel.INFO)
n, converged = solver.solve(u)
assert(converged)
print(f"Number of interations: {n:d}")

with dolfinx.io.XDMFFile(mesh2d.comm, "out_poisson/poisson.xdmf", "w") as file:
    file.write_mesh(mesh2d)
    file.write_function(u)

grad_u = ufl.grad(u)
area_left_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds(1)))
area_right_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds(2)))
i_left_cc = (1/area_left_cc) * dolfinx.fem.assemble_scalar(dolfinx.fem.form(kappa * ufl.sqrt(ufl.inner(grad_u, grad_u)) * ds(1)))
i_right_cc = (1/area_right_cc) * dolfinx.fem.assemble_scalar(dolfinx.fem.form(kappa * ufl.sqrt(ufl.inner(grad_u, grad_u)) * ds(2)))
print(i_left_cc, i_right_cc)