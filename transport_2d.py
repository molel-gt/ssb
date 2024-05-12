import sys

import numpy as np

import dolfinx
import ufl

from mpi4py import MPI
from petsc4py import PETSc

Ly = int(sys.argv[1])
meshname = sys.argv[2]
coverage = float(sys.argv[3])
Lx = int(sys.argv[4])
lower_cov = 0.5 * (1 - coverage) * Lx
upper_cov = Lx - 0.5 * (1 - coverage) * Lx

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{meshname}.xdmf", "r") as infile3:
        msh = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
        ct = infile3.read_meshtags(msh, name="Grid")

msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim-1)

Q = dolfinx.fem.functionspace(msh, ("DG", 0))
kappa = 1.0
# kappa = dolfinx.fem.Function(Q)
# lithium_cells = ct.indices[np.argwhere(ct.values == 1)]
# kappa.x.array[lithium_cells] = np.full_like(lithium_cells, 1, dtype=PETSc.ScalarType)
# electrolyte_cells = ct.indices[np.argwhere(ct.values == 2)]
# kappa.x.array[electrolyte_cells]  = np.full_like(electrolyte_cells, 1, dtype=PETSc.ScalarType)

V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))

# Dirichlet BCs
u0 = dolfinx.fem.Function(V)
with u0.vector.localForm() as u0_loc:
    u0_loc.set(1)

u1 = dolfinx.fem.Function(V)
with u1.vector.localForm() as u1_loc:
    u1_loc.set(0)
partially_insulated = lambda x: np.logical_and(np.isclose(x[1], 0.0), np.logical_and(lower_cov <= x[0],  x[0] <= upper_cov))
x0facet = dolfinx.mesh.locate_entities_boundary(msh, 0, partially_insulated)
# x0facet = dolfinx.mesh.locate_entities_boundary(msh, 0, lambda x: np.isclose(x[1], 0.0))
x1facet = dolfinx.mesh.locate_entities_boundary(msh, 0,
                                lambda x: np.isclose(x[1], Ly))
x0bc = dolfinx.fem.dirichletbc(u0, dolfinx.fem.locate_dofs_topological(V, 0, x0facet))
x1bc = dolfinx.fem.dirichletbc(u1, dolfinx.fem.locate_dofs_topological(V, 0, x1facet))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = dolfinx.fem.Constant(msh, PETSc.ScalarType(0))
g = dolfinx.fem.Constant(msh, PETSc.ScalarType(0))

a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ufl.ds

problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[x0bc, x1bc], petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_atol": 1.0e-12, "ksp_rtol": 1.0e-12})
uh = problem.solve()

with dolfinx.io.XDMFFile(msh.comm, f"{meshname}-{coverage}-potential.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)
grad_u = ufl.grad(uh)

W = dolfinx.fem.functionspace(msh, ("Lagrange", 1))

current_expr = dolfinx.fem.Expression(ufl.sqrt(ufl.inner(grad_u, grad_u)), W.element.interpolation_points)
current_h = dolfinx.fem.Function(W)
current_h.interpolate(current_expr)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{meshname}-{coverage}-current.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(current_h)
