import dolfinx
import numpy as np
import pyvista

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
mesh = create_unit_square(comm, 10, 10)

# u_ex = Constant(mesh, ScalarType(0.0))
u_ex = lambda x: 1 + x[0]**2 + 2*x[1]**2
# u_ex = lambda x: 0
x = SpatialCoordinate(mesh)
# Define physical parameters and boundary condtions
s = Constant(mesh, ScalarType(0.005))  # u_ex(x)
f = Constant(mesh, ScalarType(0.0))  # -div(grad(u_ex(x)))
n = FacetNormal(mesh)
g = Constant(mesh, ScalarType(0.0))  # -dot(n, grad(u_ex(x)))
kappa = Constant(mesh, ScalarType(1))
r = Constant(mesh, ScalarType(i0 * z * F_farad / (R * T)))  # Constant(mesh, ScalarType(1000))
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

class BoundaryCondition():
    def __init__(self, type, marker, values):
        self._type = type
        if type == "Dirichlet":
            u_D = Function(V)
            # u_D.interpolate(values)
            facets = facet_tag.find(marker)
            dofs = locate_dofs_topological(V, fdim, facets)
            with u_D.vector.localForm() as u0_loc:
                u0_loc.set(1.0)
            self._bc = dirichletbc(u_D, dofs)
        elif type == "Neumann":
                self._bc = inner(values, v) * ds(marker)
        elif type == "Robin":
            self._bc = values[0] * inner(u - values[1], v)* ds(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type

# Define the Dirichlet condition
boundary_conditions = [BoundaryCondition("Dirichlet", 1, u_ex),
                        BoundaryCondition("Robin", 2, (r, s)),
                       BoundaryCondition("Neumann", 3, g),
                       BoundaryCondition("Neumann", 4, g)]

bcs = []
for condition in boundary_conditions:
    if condition.type == "Dirichlet":
        bcs.append(condition.bc)
    else:
        F += condition.bc

# Solve linear variational problem
options = {"ksp_type": "preonly", "pc_type": "lu"}
a = lhs(F)
L = rhs(F)
problem = LinearProblem(a, L, bcs=bcs, petsc_options=options)
uh = problem.solve()

# save to file
with dolfinx.io.XDMFFile(comm, "secondary.xdmf", "w") as outfile:
    outfile.write_mesh(mesh)
    outfile.write_function(uh)
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