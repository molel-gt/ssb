#!/usr/bin/env python3

import sys

import dolfinx
import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc

sys.path.append("../")

from ssb import commons

markers = commons.SurfaceMarkers()
Lx = 2
Ly = 2

mesh2d = dolfinx.mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (Lx, Ly)), n=(16, 16),
                            cell_type=dolfinx.mesh.CellType.triangle,)
x = ufl.SpatialCoordinate(mesh2d)
n = ufl.FacetNormal(mesh2d)

K0 = dolfinx.fem.Constant(mesh2d, PETSc.ScalarType(0.1))  # siemens per meter --> bulk conductivity
i0 = dolfinx.fem.Constant(mesh2d, PETSc.ScalarType(10))  # amperes per square meter --> exchange current density
R = dolfinx.fem.Constant(mesh2d, PETSc.ScalarType(8.314))  # joules per mole per kelvin--> Universal gas constant
T = dolfinx.fem.Constant(mesh2d, PETSc.ScalarType(298))  # kelvin
alpha_a = dolfinx.fem.Constant(mesh2d, PETSc.ScalarType(0.5))
alpha_c = dolfinx.fem.Constant(mesh2d, PETSc.ScalarType(0.5))
F = dolfinx.fem.Constant(mesh2d, PETSc.ScalarType(96485))  # coulomb per mole --> Faraday constant
u_am = dolfinx.fem.Constant(mesh2d, PETSc.ScalarType(0))

V = dolfinx.fem.FunctionSpace(mesh2d, ("CG", 1))
v = ufl.TestFunction(V)
u = dolfinx.fem.Function(V)

f = dolfinx.fem.Constant(mesh2d, PETSc.ScalarType(0.0))
g = dolfinx.fem.Constant(mesh2d, PETSc.ScalarType(0.0))

# boundary conditions
left_cc_facets = dolfinx.mesh.locate_entities_boundary(mesh2d, dim=1, marker=lambda x: np.isclose(x[0], 0.0))
right_cc_facets = dolfinx.mesh.locate_entities_boundary(mesh2d, dim=1, marker=lambda x: np.isclose(x[0], Lx))
insulated_facet = dolfinx.mesh.locate_entities_boundary(mesh2d, 1, lambda x: np.logical_and(np.logical_not(np.isclose(x[0], 0)), np.logical_not(np.isclose(x[0], Lx))))

left_cc_dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=left_cc_facets)
right_cc_dofs = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=1, entities=right_cc_facets)

facets_ct_indices = np.hstack((left_cc_facets, right_cc_facets, insulated_facet))
facets_ct_values = np.hstack((markers.left_cc * np.ones(left_cc_facets.shape[0], dtype=np.int32), markers.right_cc * np.ones(right_cc_facets.shape[0], dtype=np.int32),
                            markers.insulated * np.ones(insulated_facet.shape[0], dtype=np.int32)))
facets_ct = commons.Facet(facets_ct_indices, facets_ct_values)
surf_meshtags = dolfinx.mesh.meshtags(mesh2d, 1, facets_ct.indices, facets_ct.values)

ds = ufl.Measure("ds", domain=mesh2d, subdomain_data=surf_meshtags, subdomain_id=markers.insulated)
ds_left_cc = ufl.Measure('ds', domain=mesh2d, subdomain_data=surf_meshtags, subdomain_id=markers.left_cc)
ds_right_cc = ufl.Measure('ds', domain=mesh2d, subdomain_data=surf_meshtags, subdomain_id=markers.right_cc)

F = ufl.inner(K0 * ufl.grad(u), ufl.grad(v)) * ufl.dx

# set bcs
W = dolfinx.fem.FunctionSpace(mesh2d, ("Lagrange", 1))

def grad_u_bv(u):
    """magnitude of gradient of potential expressed using Butler-Volmer
    for where there is galvanostatic cycling
    """
    return -(i0/K0) * (ufl.exp(alpha_a * F * (u - u_am) / R / T) - ufl.exp(-alpha_c * F * (u - u_am)) / R / T)

g_left_cc = dolfinx.fem.Expression(grad_u_bv(u), W.interpolation_points)
left_cc_bc = dolfinx.fem.dirichletbc(V, g_left_cc, dofs=left_cc_dofs)
# right_cc = dolfinx.fem.dirichletbc(value=PETSc.ScalarType(0), dofs=right_cc_dofs, V=V)
# bcs = [left_cc, right_cc]
options = {
               "ksp_type": "gmres",
               "pc_type": "hypre",
               "ksp_rtol": 1.0e-12
               }
a = ufl.inner(K0 * ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ds
model = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[left_cc_bc], petsc_options=options)
u = model.solve()
# Create nonlinear problem and Newton solver
# problem = dolfinx.fem.petsc.NonlinearProblem(F, u)
# solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
# solver.convergence_criterion = "incremental"
# solver.rtol = 1e-6

# ksp = solver.krylov_solver
# opts = PETSc.Options()
# option_prefix = ksp.getOptionsPrefix()
# opts[f"{option_prefix}ksp_type"] = "gmres"
# opts[f"{option_prefix}pc_type"] = "hypre"
# opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
# ksp.setFromOptions()
# r = solver.solve(u)

with dolfinx.io.XDMFFile(mesh2d.comm, "out_poisson/poisson.xdmf", "w") as file:
    file.write_mesh(mesh2d)
    file.write_function(u)