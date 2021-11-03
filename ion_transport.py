
import dolfinx
import numpy as np
import ufl
from dolfinx import (DirichletBC, Function, FunctionSpace, fem,
                     plot, BoxMesh
                     )
from dolfinx.fem import locate_dofs_topological
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from dolfinx.cpp.mesh import CellType
from mpi4py import MPI
from petsc4py import PETSc
from ufl import cos, ds, dx, exp, grad, inner, pi, sin


with XDMFFile(MPI.COMM_WORLD, "mesh_tetr.xdmf", "r") as infile3:
    mesh = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.shared_facet, 'Grid')
print("done loading tetrahedral mesh")

# with XDMFFile(MPI.COMM_WORLD, "mesh_tria.xdmf", "r") as infile2:
#     mesh_2d = infile2.read_mesh(dolfinx.cpp.mesh.GhostMode.shared_facet, "Grid")
# print("done reading triangle mesh")
Lx = 30  # side length for cube
V = FunctionSpace(mesh, ("Lagrange", 2))

# Define boundary condition on x = 0 or x = 1
u0 = Function(V)
with u0.vector.localForm() as u0_loc:
    u0_loc.set(1)

u1 = Function(V)
with u1.vector.localForm() as u1_loc:
    u1_loc.set(0)
x0facet = locate_entities_boundary(mesh, 2,
                                   lambda x: np.isclose(x[0], 0.0))
x1facet = locate_entities_boundary(mesh, 2,
                                   lambda x: np.isclose(x[0], Lx))
# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(mesh)
f = 0

x0bc = DirichletBC(u0, locate_dofs_topological(V, 2, x0facet))
x1bc = DirichletBC(u1, locate_dofs_topological(V, 2, x1facet))

g = sin(2*pi*x[1]/Lx) * sin(2*pi*x[2]/Lx)
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx(x) + inner(g, v) * ds(mesh)

print("setting problem..")

problem = fem.LinearProblem(a, L, bcs=[x0bc, x1bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# When we want to compute the solution to the problem, we can specify
# what kind of solver we want to use.
print('solving problem..')
uh = problem.solve()

# Save solution in XDMF format
with XDMFFile(MPI.COMM_WORLD, "ion_transport.xdmf", "w") as outfile:
    outfile.write_mesh(mesh)
    outfile.write_function(uh)

# Update ghost entries and plot
uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)