#!/usr/bin/env python3

from pyclbr import Function
import dolfinx
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner

from mpi4py import MPI
from petsc4py import PETSc


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
mesh2d = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (Lx, Ly)), n=(16, 16),
                            cell_type=mesh.CellType.triangle,)
V = fem.FunctionSpace(mesh2d, ("Lagrange", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
# u = fem.Function(V)
# P1 = ufl.FiniteElement("Lagrange", mesh2d.ufl_cell(), 1)
# ME = fem.FunctionSpace(mesh2d, P1 * P1)
# q, v = ufl.TestFunctions(ME)
# u = ufl.TrialFunction(ME)
# phi, i = ufl.split(u)
# print(u.x)
# u0 = ufl.variable(u)
x = ufl.SpatialCoordinate(mesh2d)
f = dolfinx.fem.Constant(mesh2d, PETSc.ScalarType(0.0))
g = dolfinx.fem.Constant(mesh2d, PETSc.ScalarType(0.0))
a = inner(K0 * grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds
# left_cc_value = (-i0 / K0) * (ufl.exp(alpha_a * F * (u0 - u_am) / (R * T)) - ufl.exp(-alpha_c * F * (u - u_am) / (R * T)))
# right_cc_value = (-i0 / K0) * (ufl.exp(alpha_a * F * (u0 - u_am) / (R * T)) - ufl.exp(-alpha_c * F * (u - u_am) / (R * T)))
# u.x.array[0, :, :]
# boundary conditions
# print(u.x)
left_cc_facets = mesh.locate_entities_boundary(mesh2d, dim=1, marker=lambda x: np.isclose(x[0], 0.0))
right_cc_facets = mesh.locate_entities_boundary(mesh2d, dim=1, marker=lambda x: np.isclose(x[0], Lx))
left_cc_dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=left_cc_facets)
right_cc_dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=right_cc_facets)

# set bcs
left_cc = fem.dirichletbc(value=PETSc.ScalarType(1), dofs=left_cc_dofs, V=V)
right_cc = fem.dirichletbc(value=PETSc.ScalarType(0), dofs=right_cc_dofs, V=V)

problem = fem.petsc.LinearProblem(a, L, bcs=[left_cc, right_cc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

with io.XDMFFile(mesh2d.comm, "out_poisson/poisson.xdmf", "w") as file:
    file.write_mesh(mesh2d)
    file.write_function(uh)
try:
    import pyvista
    cells, types, x = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    plotter.show()
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
