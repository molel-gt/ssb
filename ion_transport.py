
import dolfinx
import numpy as np
import ufl
from dolfinx import (DirichletBC, Function, FunctionSpace, fem,
                     BoxMesh, RectangleMesh, plot
                     )
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import locate_dofs_topological
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
from petsc4py import PETSc
from ufl import ds, dx, grad, inner


# # Create mesh and define function space
# mesh = BoxMesh(
#     MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([1, 1, 1])],
#     [10, 10, 10], CellType.tetrahedron, dolfinx.cpp.mesh.GhostMode.none)
# mesh_2d = RectangleMesh(
#     MPI.COMM_WORLD,
#     [np.array([0, 0, 0]), np.array([1, 1, 0])], [10, 10],
#     CellType.triangle, dolfinx.cpp.mesh.GhostMode.none)

with XDMFFile(MPI.COMM_WORLD, "mesh_tetr.xdmf", "r") as infile3:
    mesh = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
print("done loading tetrahedral mesh")

with XDMFFile(MPI.COMM_WORLD, "mesh_tria.xdmf", "r") as infile2:
    mesh_2d = infile2.read_mesh(dolfinx.cpp.mesh.GhostMode.none, "Grid")
print("done reading triangle mesh")

V = FunctionSpace(mesh, ("Lagrange", 2))

# Define boundary condition on x = 0 or x = 1
u0 = Function(V)
with u0.vector.localForm() as u0_loc:
    u0_loc.set(0)
u1 = Function(V)

with u1.vector.localForm() as u1_loc:
    u1_loc.set(1)
x0facet = locate_entities_boundary(mesh, 2,
                                   lambda x: np.isclose(x[0], 0))
x1facet = locate_entities_boundary(mesh, 2,
                                   lambda x: np.isclose(x[0], 10))
# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(mesh)
f = 0

x0bc = DirichletBC(u0, locate_dofs_topological(V, 2, x0facet))
x1bc = DirichletBC(u1, locate_dofs_topological(V, 2, x1facet))

g = x[1] * (10 - x[1]) * x[2] * (10 - x[2])
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx(x) + inner(g, v) * ds(mesh)
print("setting problem..")

problem = fem.LinearProblem(a, L, bcs=[x0bc, x1bc],
                            petsc_options={"ksp_type": "preonly",
                            "pc_type": "lu"})

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
try:
    import pyvista

    topology, cell_types = plot.create_vtk_topology(mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)
    grid.point_data["u"] = uh.compute_point_values().real
    grid.set_active_scalars("u")

    plotter = pyvista.Plotter()
    plotter.enable_depth_peeling(10)
    plotter.add_mesh(grid, color=True)
    # plotter.add_mesh(grid.copy(), style="points", render_points_as_spheres=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)

    # If pyvista environment variable is set to off-screen (static) plotting save png
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot("uh.png")
    else:
        plotter.show()
except ModuleNotFoundError:
    print("pyvista is required to visualise the solution")
