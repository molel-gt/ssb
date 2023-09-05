import dolfinx
import numpy as np
import pyvista
import ufl

from dolfinx import cpp, fem, io, mesh, plot
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 div, dx, ds, grad, inner, grad)

comm = MPI.COMM_WORLD
N = 8
domain = mesh.create_unit_square(comm, N, N)
V = fem.FunctionSpace(domain, ("CG", 1))

uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)
uD.x.scatter_forward()
x = ufl.SpatialCoordinate(domain)
n = ufl.FacetNormal(domain)

f = -div(grad(1 + x[0] ** 2 + 2 * x[1] ** 2))

u = TrialFunction(V)
v = TestFunction(V)
h = 2 * Circumradius(domain)
alpha = 10
a = inner(grad(u), grad(v)) * dx - inner(n, grad(u)) * v * ds
a += - inner(n, grad(v)) * u * ds + alpha / h * inner(u, v) * ds
L = inner(f, v) * dx
L += - inner(n, grad(v)) * uD * ds + alpha / h * inner(uD, v) * ds

problem = fem.petsc.LinearProblem(a, L)
uh = problem.solve()

with io.XDMFFile(comm, "nitsche.xdmf", "w") as outfile:
    outfile.write_mesh(domain)
    outfile.write_function(uh)

error_form = fem.form(inner(uh - uD, uh - uD) * dx)
errorL2 = np.sqrt(fem.assemble_scalar(error_form))
print(fr"$L^2$-error: {errorL2:.2e}")

u_vertex_values = uh.x.array.real
u_ex_vertex_values = uD.x.array.real
error_max = np.max(np.abs(u_vertex_values - u_ex_vertex_values))
print(f"Error_max : {error_max:.2e}")

# topology, cell_types, x = plot.create_vtk_mesh(V)
# grid = pyvista.UnstructuredGrid(topology, cell_types, x)
# grid.point_data["u"] = uh.x.array.real
# grid.set_active_scalars("u")
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
# warped = grid.warp_by_scalar()
# plotter.add_mesh(warped)
# plotter.view_xy()
# if not pyvista.OFF_SCREEN:
#     pyvista.start_xvfb(wait=0.05)
#     plotter.show()
# figure = plotter.screenshot("nitsche.png")

