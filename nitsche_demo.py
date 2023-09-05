import dolfinx
import dolfinx.plot
import numpy
import pyvista
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 div, dx, ds, grad, inner, grad)

N = 8
mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N)
V = dolfinx.FunctionSpace(mesh, ("CG", 1))

uD = dolfinx.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
dolfinx.cpp.la.scatter_forward(uD.x)
x = SpatialCoordinate(mesh)
f = -div(grad(1 + x[0]**2 + 2 * x[1]**2))

u = TrialFunction(V)
v = TestFunction(V)
n = FacetNormal(mesh)
h = 2 * Circumradius(mesh)
alpha = 10
a = inner(grad(u), grad(v)) * dx - inner(n, grad(u)) * v * ds
a += - inner(n, grad(v)) * u * ds + alpha / h * inner(u, v) * ds
L = inner(f, v) * dx
L += - inner(n, grad(v)) * uD * ds + alpha / h * inner(uD, v) * ds

problem = dolfinx.fem.LinearProblem(a, L)
uh = problem.solve()

error_form = inner(uh-uD, uh-uD) * dx
errorL2 = numpy.sqrt(dolfinx.fem.assemble_scalar(error_form))
print(fr"$L^2$-error: {errorL2:.2e}")

u_vertex_values = uh.compute_point_values()
u_ex_vertex_values = uD.compute_point_values()
error_max = numpy.max(numpy.abs(u_vertex_values - u_ex_vertex_values))
print(f"Error_max : {error_max:.2e}")

grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)
grid.point_arrays["u"] = u_vertex_values.real
grid.set_active_scalars("u")
pyvista.start_xvfb(wait=0.05)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
figure = plotter.screenshot("nitsche.png")

