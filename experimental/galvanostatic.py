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
K0 = 0.1  # siemens per meter --> bulk conductivity
i0 = 10  # amperes per square meter --> exchange current density
R = 8.314  # joules per mole per kelvin--> Universal gas constant
T = 298  # kelvin
alpha_a = 0.5
alpha_c = 0.5
F = 96485  # coulomb per mole --> Faraday constant
u_am = 0
mesh2d = dolfinx.mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (Lx, Ly)), n=(16, 16),
                            cell_type=dolfinx.mesh.CellType.triangle,)
x = ufl.SpatialCoordinate(mesh2d)
n = ufl.FacetNormal(mesh2d)
V = dolfinx.fem.FunctionSpace(mesh2d, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

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

a = ufl.inner(K0 * ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ds + ufl.inner(g_cc, v) * ds_left_cc #+ ufl.inner(g_cc, v) * ds_right_cc

# set bcs
left_cc = fem.dirichletbc(value=PETSc.ScalarType(1), dofs=left_cc_dofs, V=V)
right_cc = dolfinx.fem.dirichletbc(value=PETSc.ScalarType(0), dofs=right_cc_dofs, V=V)
bcs = [left_cc, right_cc]
options = {
               "ksp_type": "gmres",
               "pc_type": "hypre",
               "ksp_rtol": 1.0e-12
               }
problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options=options)
uh = problem.solve()

with dolfinx.io.XDMFFile(mesh2d.comm, "out_poisson/poisson.xdmf", "w") as file:
    file.write_mesh(mesh2d)
    file.write_function(uh)
