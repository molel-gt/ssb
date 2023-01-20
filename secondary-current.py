import dolfinx
import numpy as np
import pyvista
import ufl

from dolfinx.fem import (Constant,  Function, FunctionSpace, assemble_scalar, 
                         dirichletbc, form, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square, locate_entities, meshtags
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import (FacetNormal, Measure, SpatialCoordinate, TestFunction, TrialFunction, 
                 div, dot, dx, grad, inner, lhs, rhs)
from dolfinx.io import XDMFFile
from dolfinx.plot import create_vtk_mesh


# parameters
R = 8.314 # J/K/mol
T = 298 # K
z = 1  # number of electrons involved
F_farad = 96485  # C/mol
i0 = 10  # A/m^2

comm = MPI.COMM_WORLD
mesh = create_unit_square(comm, 20, 20)

u_left_bc = 1.0
x = SpatialCoordinate(mesh)

# Define physical parameters and boundary condtions
s = Constant(mesh, ScalarType(0.005))
f = Constant(mesh, ScalarType(0.0))
n = FacetNormal(mesh)
# i0 = Constant(mesh, ScalarType(10))  # A/m^2
g = Constant(mesh, ScalarType(0.0))
kappa = Constant(mesh, ScalarType(1))
r = Constant(mesh, ScalarType(i0 * z * F_farad / (R * T)))

# Define function space and standard part of variational form
V = FunctionSpace(mesh, ("CG", 1))
u, v = TrialFunction(V), TestFunction(V)
F = kappa * inner(grad(u), grad(v)) * dx - inner(f, v) * dx

boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], 1)),
              (3, lambda x: np.isclose(x[1], 0)),
              (4, lambda x: np.isclose(x[1], 1))]

facet_indices, facet_markers = [], []
fdim = mesh.topology.dim - 1
for (marker, locator) in boundaries:
    facets = locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
with XDMFFile(comm, "facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(facet_tag)

ds = Measure("ds", domain=mesh, subdomain_data=facet_tag)

# Dirichlet boundary
u_D = Function(V)
facets = facet_tag.find(1)
dofs = locate_dofs_topological(V, fdim, facets)
with u_D.vector.localForm() as u0_loc:
    u0_loc.set(1.0)

bcs = [dirichletbc(u_D, dofs)]

# Neumann boundary
F += inner(g, v) * ds(3)
F += inner(g, v) * ds(4)

# Robin boundary
F += r * inner(u - s, v) * ds(2)

# Solve linear variational problem
options = {
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "ksp_rtol": 1.0e-12,
        }
a = lhs(F)
L = rhs(F)
problem = LinearProblem(a, L, bcs=bcs, petsc_options=options)
uh = problem.solve()

# save to file
with dolfinx.io.XDMFFile(comm, "secondary-potential.xdmf", "w") as outfile:
    outfile.write_mesh(mesh)
    outfile.write_function(uh)

grad_u = ufl.grad(uh)

W = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
current_expr = dolfinx.fem.Expression(kappa * ufl.sqrt(ufl.inner(grad_u, grad_u)), W.element.interpolation_points())
current_h = dolfinx.fem.Function(W)
current_h.interpolate(current_expr)

with dolfinx.io.XDMFFile(comm, "secondary-current.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(current_h)

# Visualize solution
pyvista.set_jupyter_backend("pythreejs")
pyvista_cells, cell_types, geometry = create_vtk_mesh(V)
grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)
grid.point_data["u"] = uh.x.array
grid.set_active_scalars("u")

plotter = pyvista.Plotter()
plotter.add_text("uh", position="upper_edge", font_size=14, color="black")
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    pyvista.start_xvfb()
    figure = plotter.screenshot("robin_neumann_dirichlet.png")