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


pyvista.set_plot_theme("paraview")

# parameters
R = 8.314 # J/K/mol
T = 298 # K
z = 1  # number of electrons involved
F_farad = 96485  # C/mol
i0 = 10  # A/m^2

plotter = pyvista.Plotter()

comm = MPI.COMM_WORLD
mesh = create_unit_square(comm, 25, 25)


def create_mesh(kappa=0.5, y_lower=0.25, y_upper=0.75):
    """Wrapper to allow value update on parameter change"""
    u_left_bc = 1.0
    x = SpatialCoordinate(mesh)

    # Define physical parameters and boundary conditions
    s = Constant(mesh, ScalarType(0.005))
    f = Constant(mesh, ScalarType(0.0))
    n = FacetNormal(mesh)
    # i0 = Constant(mesh, ScalarType(10))  # A/m^2
    g = Constant(mesh, ScalarType(0.0))
    kappa = Constant(mesh, ScalarType(kappa))
    r = Constant(mesh, ScalarType(i0 * z * F_farad / (R * T)))

    # Define function space and standard part of variational form
    V = FunctionSpace(mesh, ("CG", 1))
    u, v = TrialFunction(V), TestFunction(V)
    F = kappa * inner(grad(u), grad(v)) * dx - inner(f, v) * dx

    # boundaries
    left_cc = lambda x: np.logical_and(np.isclose(x[0], 0), np.logical_and(np.less_equal(x[1], y_upper), np.greater_equal(x[1], y_lower)))
    right_cc = lambda x: np.isclose(x[0], 1.0)
    insulated = lambda x: np.logical_and(
        np.logical_not(np.logical_and(np.isclose(x[0], 0), np.logical_and(np.less_equal(x[1], y_upper), np.greater_equal(x[1], y_lower)))),
        np.logical_not(np.isclose(x[0], 1.0)),
    )

    boundaries = [
        (1, left_cc),
        (2, right_cc),
        (3, insulated),
    ]
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

    # Dirichlet boundary - set potential
    u_D = Function(V)
    facets = facet_tag.find(2)
    dofs = locate_dofs_topological(V, fdim, facets)
    with u_D.vector.localForm() as u0_loc:
        u0_loc.set(1.0)

    bcs = [dirichletbc(u_D, dofs)]

    # Neumann boundary - insulated
    F += inner(g, v) * ds(3)

    # Robin boundary - variable area - set kinetics expression
    F += r * inner(u - s, v) * ds(1)

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


    plotter.add_text("potential", position="lower_edge", font_size=14, color="black")
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()


class MyCustomRoutine:
    def __init__(self, mesh):
        self.output = mesh
        self.kwargs = {
            'kappa': 0.5,
            'y_lower': 0.25,
            'y_upper': 0.75,
        }

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def update(self):
        # This is where you call your simulation
        create_mesh(**self.kwargs)
        return

engine = MyCustomRoutine(mesh)

plotter.add_slider_widget(
    callback=lambda value: engine('kappa', value),
    rng=[0, 10],
    value=0.5,
    title="conductivity",
    pointa=(0.025, 0.925),
    pointb=(0.31, 0.925),
    style='modern',
)
plotter.add_slider_widget(
    callback=lambda value: engine('y_lower', value),
    rng=[0, 0.5],
    value=0.25,
    title="min y",
    pointa=(0.35, 0.925),
    pointb=(0.64, 0.925),
    style='modern',
)
plotter.add_slider_widget(
    callback=lambda value: engine('y_upper', value),
    rng=[0.5, 1],
    value=0.75,
    title="max y",
    pointa=(0.70, 0.925),
    pointb=(0.99, 0.925),
    style='modern',
)
plotter.show()