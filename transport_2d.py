import sys

import numpy as np

import dolfinx
import ufl

from mpi4py import MPI
from petsc4py import PETSc

Ly = int(sys.argv[1])

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as infile3:
        msh = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')

V = dolfinx.fem.FunctionSpace(msh, ("Lagrange", 1))

# Dirichlet BCs
u0 = dolfinx.fem.Function(V)
with u0.vector.localForm() as u0_loc:
    u0_loc.set(1)

u1 = dolfinx.fem.Function(V)
with u1.vector.localForm() as u1_loc:
    u1_loc.set(0)
x0facet = dolfinx.mesh.locate_entities_boundary(msh, 0,
                                lambda x: np.isclose(x[1], 0.0))
x1facet = dolfinx.mesh.locate_entities_boundary(msh, 0,
                                lambda x: np.isclose(x[1], Ly))
x0bc = dolfinx.fem.dirichletbc(u0, dolfinx.fem.locate_dofs_topological(V, 0, x0facet))
x1bc = dolfinx.fem.dirichletbc(u1, dolfinx.fem.locate_dofs_topological(V, 0, x1facet))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = dolfinx.fem.Constant(msh, PETSc.ScalarType(0))
g = dolfinx.fem.Constant(msh, PETSc.ScalarType(0))

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ufl.ds

problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[x0bc, x1bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

with dolfinx.io.XDMFFile(msh.comm, "potential.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)
grad_u = ufl.grad(uh)

W = dolfinx.fem.FunctionSpace(msh, ("Lagrange", 1))
current_expr = dolfinx.fem.Expression(ufl.sqrt(ufl.inner(grad_u, grad_u)), W.element.interpolation_points)
current_h = dolfinx.fem.Function(W)
current_h.interpolate(current_expr)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "current.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(current_h)