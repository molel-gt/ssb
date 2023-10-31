from dolfinx import mesh, fem, plot, geometry, io
import dolfinx.fem.petsc
import dolfinx.nls.petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc
from ufl import dx, grad, dot, jump, avg
import numpy as np
import matplotlib.pyplot as plt
import pyvista

comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 100, 100)
V = fem.FunctionSpace(domain, ("DG", 1))
uD = fem.Function(V)
uD.interpolate(lambda x: np.full(x[0].shape, 0.0))

# create mesh tags
def marker_interface(x):
    return np.isclose(x[0], 0.5)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
facet_imap = domain.topology.index_map(tdim - 1)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
interface_facets = mesh.locate_entities(domain, tdim - 1, marker_interface)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
indices = np.arange(0, num_facets)
# values = np.arange(0, num_facets, dtype=np.intc)
values = np.zeros(indices.shape, dtype=np.intc)  # all facets are tagged with zero

values[boundary_facets] = 1
values[interface_facets] = 2

mesh_tags_facets = mesh.meshtags(domain, tdim - 1, indices, values)

ds = ufl.Measure("ds", domain=domain, subdomain_data=mesh_tags_facets)
dS = ufl.Measure("dS", domain=domain, subdomain_data=mesh_tags_facets)
u = fem.Function(V)
u_n = fem.Function(V)
v = ufl.TestFunction(V)

h = ufl.CellDiameter(domain)
n = ufl.FacetNormal(domain)

# Define parameters
alpha = 1000

# Simulation constants
f = fem.Constant(domain, PETSc.ScalarType(2.0))
K1 = fem.Constant(domain, PETSc.ScalarType(2.0))
K2 = fem.Constant(domain, PETSc.ScalarType(4.0))

# Define variational problem
F = 0
F += dot(grad(v), grad(u)) * dx - dot(v * n, grad(u)) * ds
F += - dot(avg(grad(v)), jump(u, n)) * dS(0)
F += - dot(jump(v, n), avg(grad(u))) * dS(0)
F += + alpha/avg(h) * dot(jump(v, n), jump(u, n)) * dS(0)
F += + alpha/h * v * u * ds

# source
F += -v * f * dx

# Dirichlet BC
F += - dot(grad(v), u * n) * ds
F += + uD * dot(grad(v), n) * ds - alpha/h * uD * v * ds

# Interface
F += - dot(avg(grad(v)), n('-')) * (u('-') * (K1/K2-1)) * dS(2)
F += alpha/avg(h) * dot(jump(v,n), n('-')) * (u('-') * (K1/K2-1)) * dS(2)

# symmetry
F += - dot(avg(grad(v)), jump(u, n)) * dS(2)

# coercivity
F += + alpha/avg(h) * dot(jump(v, n), jump(u, n)) * dS(2)

problem = dolfinx.fem.petsc.NonlinearProblem(F, u)
solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
solver.solve(u)
u.name = 'potential'
with io.XDMFFile(comm, 'potential-discontinuous.xdmf', 'w') as fp:
    fp.write_mesh(domain)
    fp.write_function(u)

# bb_tree = geometry.bb_tree(domain, domain.topology.dim)
# n_points = 1000
# tol = 0.001  # Avoid hitting the outside of the domain
# x = np.linspace(0 + tol, 1 - tol, n_points)
# y = np.ones(n_points) * 0.5
# points = np.zeros((3, n_points))
# points[0] = x
# points[1] = y
# u_values = []
# cells = []
# points_on_proc = []
# # Find cells whose bounding-box collide with the the points
# cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
# # Choose one of the cells that contains the point
# colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
# for i, point in enumerate(points.T):
#     if len(colliding_cells.links(i)) > 0:
#         points_on_proc.append(point)
#         cells.append(colliding_cells.links(i)[0])
# points_on_proc = np.array(points_on_proc, dtype=np.float64)
# u_values = u.eval(points_on_proc, cells)
# fig = plt.figure()
# plt.plot(points_on_proc[:, 0], u_values, "k", linewidth=2)
# plt.grid(True)
# plt.show()
#
# pyvista.OFF_SCREEN = True
#
# pyvista.start_xvfb()
#
# u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
# u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
# u_grid.point_data["u"] = u.x.array.real
# u_grid.set_active_scalars("u")
# u_plotter = pyvista.Plotter()
# u_plotter.add_mesh(u_grid, show_edges=False)
# u_plotter.view_xy()
# if not pyvista.OFF_SCREEN:
#     u_plotter.show()
# else:
#     figure = u_plotter.screenshot("DG.png")