#!/usr/bin/env python3

import sys
sys.path.append("../")

import dolfinx
import numpy as np
import ufl

from dolfinx import mesh, fem, io, nls, log

from mpi4py import MPI
from petsc4py import PETSc

from ssb import commons

markers = commons.SurfaceMarkers()


i0 = 10  # exchange current density
phi2 = 0.5
F_c = 96485  # Faraday constant
R = 8.314
T = 298
alpha_a = 0.5
alpha_c = 0.5

comm = MPI.COMM_WORLD
with io.XDMFFile(comm, "mesh/laminate/tria.xdmf", "r") as infile2:
    mesh2d = infile2.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
    ct = infile2.read_meshtags(mesh2d, name="Grid")

with io.XDMFFile(comm, "mesh/laminate/line.xdmf", "r") as infile1:
    mesh1d = infile1.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
    ft = infile1.read_meshtags(mesh1d, name="Grid")

mesh2d.topology.create_connectivity(mesh2d.topology.dim, mesh2d.topology.dim - 1)

x = ufl.SpatialCoordinate(mesh2d)

# Define physical parameters and boundary condtions
f = dolfinx.fem.Constant(mesh2d, PETSc.ScalarType(0.0))
n = ufl.FacetNormal(mesh2d)
g = dolfinx.fem.Constant(mesh2d, PETSc.ScalarType(0.0))

kappa = fem.Constant(mesh2d, PETSc.ScalarType(1))

# Define function space and standard part of variational form
V = fem.FunctionSpace(mesh2d, ("CG", 1))
u, v = fem.Function(V), ufl.TestFunction(V)

def i_butler_volmer(phi1=u, phi2=phi2):
    return i0  * (ufl.exp(alpha_a * F_c * (u - phi2) / R / T) - ufl.exp(-alpha_c * F_c * (u - phi2) / R / T))

g_curr = -i_butler_volmer() / kappa

boundaries = [(markers.left_cc, lambda x: np.isclose(x[0], 0)),
              (markers.right_cc, lambda x: np.isclose(x[0], 1)),
              (markers.insulated, lambda x: np.isclose(x[1], 0)),
              (markers.insulated, lambda x: np.isclose(x[1], 1))]

# facet_indices, facet_markers = [], []
fdim = mesh2d.topology.dim - 1
# for (marker, locator) in boundaries:
#     facets = mesh.locate_entities(mesh2d, fdim, locator)
#     facet_indices.append(facets)
#     facet_markers.append(np.full_like(facets, marker))
# facet_indices = np.hstack(facet_indices).astype(np.int32)
# facet_markers = np.hstack(facet_markers).astype(np.int32)
# sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(mesh2d, fdim, ft.indices, ft.values)

# mesh2d.topology.create_connectivity(mesh2d.topology.dim - 1, mesh2d.topology.dim)
# with io.XDMFFile(comm, "mesh/galvanostatic/facet_tags.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh2d)
#     xdmf.write_meshtags(facet_tag)

ds = ufl.Measure("ds", domain=mesh2d, subdomain_data=facet_tag)
u0 = dolfinx.fem.Function(V)
with u0.vector.localForm() as u0_loc:
    u0_loc.set(0)

x0facet = dolfinx.mesh.locate_entities_boundary(mesh2d, 1, lambda x: np.isclose(x[0], 0.0))
x0bc = dolfinx.fem.dirichletbc(u0, dolfinx.fem.locate_dofs_topological(V, 1, x0facet))

F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx - ufl.inner(g_curr, v) * ds(markers.right_cc) - ufl.inner(g, v) * ds(markers.insulated) - ufl.inner(g, v) * ds(markers.insulated)

problem = fem.petsc.NonlinearProblem(F, u, bcs=[x0bc])
solver = nls.petsc.NewtonSolver(comm, problem)
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
opts[f"{option_prefix}maximum_iterations"] = 100
ksp.setFromOptions()

log.set_log_level(log.LogLevel.WARNING)
n, converged = solver.solve(u)
assert(converged)
print(f"Number of interations: {n:d}")

with dolfinx.io.XDMFFile(comm, "mesh/galvanostatic/potential.xdmf", "w") as file:
    file.write_mesh(mesh2d)
    file.write_function(u)

grad_u = ufl.grad(u)
area_left_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds(1)))
area_right_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds(2)))
i_left_cc = (1/area_left_cc) * dolfinx.fem.assemble_scalar(dolfinx.fem.form(kappa * ufl.sqrt(ufl.inner(grad_u, grad_u)) * ds(1)))
i_right_cc = (1/area_right_cc) * dolfinx.fem.assemble_scalar(dolfinx.fem.form(kappa * ufl.sqrt(ufl.inner(grad_u, grad_u)) * ds(2)))

W = dolfinx.fem.FunctionSpace(mesh2d, ("Lagrange", 1))
current_expr = dolfinx.fem.Expression(kappa * ufl.sqrt(ufl.inner(grad_u, grad_u)), W.element.interpolation_points())
current_h = dolfinx.fem.Function(W)
current_h.interpolate(current_expr)

with dolfinx.io.XDMFFile(comm, "mesh/galvanostatic/current.xdmf", "w") as file:
    file.write_mesh(mesh2d)
    file.write_function(current_h)

print("Current density @ left cc                       : {:.4f}".format(i_left_cc))
print("Current density @ right cc                      : {:.4f}".format(i_right_cc))