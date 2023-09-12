import dolfinx
import numpy
import ufl

from dolfinx import cpp, fem, io, mesh, nls
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 div, dx, ds, dS, grad, inner, grad, avg, jump)

N = 8
domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
V = fem.FunctionSpace(domain, ("DG", 1))

V_CG = fem.FunctionSpace(domain, ("CG", 1))
print(f"Num DG dofs: {V.dofmap.index_map.size_local}")
print(f"Num CG dofs: {V_CG.dofmap.index_map.size_local}")

uD = fem.Function(V)
a = 0.8
c = 1
uD.interpolate(lambda x: 1 + a * x[0] ** 2 + c * x[1] ** 2)
uD.x.scatter_forward()
x = ufl.SpatialCoordinate(domain)
f = -div(grad(1 + a * x[0] ** 2 + c * x[1] ** 2))

u = TrialFunction(V)
v = TestFunction(V)
n = FacetNormal(domain)
h = 2 * Circumradius(domain)
alpha = 10
gamma = 10
h_avg = avg(h)

a = inner(grad(u), grad(v)) * dx - inner(n, grad(u)) * v * ds
# Add DG/IP terms
a += - inner(avg(grad(v)), jump(u, n)) * dS - inner(jump(v, n), avg(grad(u))) * dS
a += + (gamma / h_avg) * inner(jump(v, n), jump(u, n)) * dS

# Add Nitsche terms
a += - inner(n, grad(v)) * u * ds + alpha / h * inner(u, v) * ds
L = inner(f, v) * dx
L += - inner(n, grad(v)) * uD * ds + alpha / h * inner(uD, v) * ds

problem = fem.petsc.LinearProblem(a, L)
uh = problem.solve()
