
import dolfinx
import dolfin as dol
import numpy as np
import ufl
from dolfinx import (DirichletBC, Function, FunctionSpace, RectangleMesh, fem,
                     UnitCubeMesh, plot
                     )
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import locate_dofs_topological
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
from petsc4py import PETSc
from ufl import ds, dx, grad, inner


# # Create mesh and define function space
mesh = UnitCubeMesh(
    MPI.COMM_WORLD, 90, 90, 90,
    CellType.tetrahedron, dolfinx.cpp.mesh.GhostMode.none)
with XDMFFile(MPI.COMM_WORLD, "mesh_tetra.xdmf", "r") as infile:
    mesh = infile.read_mesh(MPI.COMM_WORLD, dolfinx.cpp.mesh.GhostMode.none)
print("done loading tetrahedral mesh")

boundaries = dol.MeshValueCollection("size_t", mesh, 2)
with XDMFFile(MPI.COMM_WORLD, "mesh_tria.xdmf") as infile:
    boundaries = infile.read_mvc_size_t(mesh, "all_tags")
print("done reading triangle mesh")
V = FunctionSpace(mesh, ("Lagrange", 1))

# Define boundary condition on x = 0 or x = 1
u0 = Function(V)
with u0.vector.localForm() as u0_loc:
    u0_loc.set(0)
u1 = Function(V)
with u1.vector.localForm() as u1_loc:
    u1_loc.set(1)
x0facet = locate_entities_boundary(boundaries, 2,
                                   lambda x: np.isclose(x[0], 0.0))
x1facet = locate_entities_boundary(boundaries, 2,
                                   lambda x: np.isclose(x[0], 1.0))
x0bc = DirichletBC(u0, locate_dofs_topological(V, 2, x0facet))
x1bc = DirichletBC(u1, locate_dofs_topological(V, 2, x1facet))

# Define variational problem
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
x = ufl.SpatialCoordinate(mesh)
f = 0
g = x[1] * (1 - x[1]) * x[2] * (1 - x[2])
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds

problem = fem.LinearProblem(a, L, bcs=[x0bc, x1bc],
                            petsc_options={"ksp_type": "preonly",
                            "pc_type": "lu"})

# When we want to compute the solution to the problem, we can specify
# what kind of solver we want to use.
uh = problem.solve()

# Save solution in XDMF format
with XDMFFile(MPI.COMM_WORLD, "ion_transport.xdmf", "w") as outfile:
    outfile.write_mesh(mesh)
    outfile.write_function(uh)
