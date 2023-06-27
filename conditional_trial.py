import dolfinx
from dolfinx import fem
from dolfinx import mesh
from dolfinx.fem import (Function, FunctionSpace, VectorFunctionSpace)
from petsc4py import PETSc
from mpi4py import MPI
import ufl as ufl
import numpy as np
from dolfinx.io import XDMFFile

L = 100.0
H = 100.0
dL = 30
dH = 30
domain = mesh.create_box(MPI.COMM_WORLD,[[0,0,0], [L, L, H]], [dL, dL, dH], mesh.CellType.tetrahedron)
x = ufl.SpatialCoordinate(domain)
n = ufl.FacetNormal(domain)

# For plasticity problem
V_pl = VectorFunctionSpace(domain, ('Lagrange', 1))
pl = Function(V_pl)
with pl.vector.localForm() as pl_loc:
    pl_loc.set(-1.0)

x0_func = Function(V_pl)
x0_ufl = x[0]
print(str(x0_ufl))
# def evltr(x):

x0 = lambda x: x[0] / x[0]
x0_func.interpolate(x0)
# print(x0_func.vector.array)
# print("min of x0_func:", min(x0_func.vector.array), flush=True)
with XDMFFile(MPI.COMM_WORLD, "test.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(x0_func)


# def project(v, target_func, bcs=[]):
#     # Ensure we have a mesh and attach to measure
#     V = target_func.function_space
#     dx = ufl.dx(V.mesh)
#
#     # Define variational problem for projection
#     w = ufl.TestFunction(V)
#     Pv = ufl.TrialFunction(V)
#     a = dolfinx.fem.form(ufl.inner(Pv, w) * dx)
#     L = dolfinx.fem.form(ufl.inner(v, w) * dx)
#
#     # Assemble linear system
#     A = dolfinx.fem.petsc.assemble_matrix(a, bcs)
#     A.assemble()
#     b = dolfinx.fem.petsc.assemble_vector(L)
#     dolfinx.fem.petsc.apply_lifting(b, [a], [bcs])
#     b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
#     dolfinx.fem.petsc.set_bc(b, bcs)
#
#     # Solve linear system
#     solver = PETSc.KSP().create(A.getComm())
#     solver.setOperators(A)
#     solver.solve(b, target_func.vector)


def Pl_max(pl_, pln_):
    return ufl.conditional(ufl.gt(pl_, pln_), pl_, pln_)

cond = Pl_max(ufl.inner(pl, n), x0_func)
print(fem.assemble_scalar(fem.form(cond * ufl.ds)))
# project(cond, pl)
# print("min of plasticity:", min(pl.vector.array), flush=True)