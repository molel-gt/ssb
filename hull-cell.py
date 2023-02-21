#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:57:07 2023

@author: molel
"""
import gmsh
import dolfinx
import meshio
import numpy as np
import pygmsh
import pyvista
import ufl

from dolfinx.fem import (Constant, FunctionSpace)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import meshtags
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import (Measure, TestFunction, TrialFunction, 
                 dx, grad, inner, lhs, rhs)
from dolfinx.plot import create_vtk_mesh


plotter = pyvista.Plotter()
comm = MPI.COMM_WORLD

resolution = 0.1

# Channel parameters
L1 = 3
L2 = 2
L = L1 + L2
W = 5

insulated_marker = 2
left_cc_marker = 3
right_cc_marker = 4

# starting
c = [2.5, 2.5, 0]
r = 0.5

### PARAMETERS ##############
# parameters
R = 8.314 # J/K/mol
T = 298 # K
z = 1  # number of electrons involved
F_farad = 96485  # C/mol
i0 = 10  # A/m^2


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh


def create_geometry(c, r):
    # Initialize empty geometry using the build in kernel in GMSH
    geometry = pygmsh.geo.Geometry()
    
    # Fetch model we would like to add data to
    model = geometry.__enter__()
    
    # Add circle
    circle = model.add_circle(c, r, mesh_size=resolution)
    
    
    points = [model.add_point((0, 0, 0), mesh_size=resolution),
              model.add_point((0, W, 0), mesh_size=resolution),
              model.add_point((L1, W, 0), mesh_size=resolution),
              model.add_point((L, 0, 0), mesh_size=resolution)]
    
    channel_lines = [model.add_line(points[i], points[i+1])
                     for i in range(-1, len(points)-1)]
    channel_loop = model.add_curve_loop(channel_lines)
    plane_surface = model.add_plane_surface(channel_loop, holes=[circle.curve_loop])
    
    # Call gmsh kernel before add physical entities
    model.synchronize()

    model.add_physical([plane_surface], "domain")
    model.add_physical([channel_lines[0], channel_lines[2]], "insulated")
    model.add_physical([channel_lines[1]], "left")
    model.add_physical([channel_lines[3]], "right")

    geometry.generate_mesh(dim=2)
    gmsh.write("mesh.msh")
    gmsh.clear()
    geometry.__exit__()

    mesh_from_file = meshio.read("mesh.msh")
    line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
    meshio.write("facet_mesh.xdmf", line_mesh)

    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    meshio.write("mesh.xdmf", triangle_mesh)


######################## MODEL ####################################
def run_model(c=c, r=r, kappa=0.5):
    """Wrapper to allow value update on parameter change"""
    create_geometry(c, r)
    with dolfinx.io.XDMFFile(comm, "mesh.xdmf", "r") as infile2:
        mesh = infile2.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

    with dolfinx.io.XDMFFile(comm, 'facet_mesh.xdmf', "r") as xdmf:
        ft = xdmf.read_meshtags(mesh, name="Grid")
    
    ft_tag = meshtags(mesh, mesh.topology.dim - 1, ft.indices, ft.values)

    # Define physical parameters and boundary conditions
    s = Constant(mesh, ScalarType(0.005))
    f = Constant(mesh, ScalarType(0.0))
   
    ds = Measure("ds", domain=mesh, subdomain_data=ft_tag)

    g = Constant(mesh, ScalarType(0.0))
    kappa = Constant(mesh, ScalarType(kappa))
    r = Constant(mesh, ScalarType(i0 * z * F_farad / (R * T)))

    # Define function space and standard part of variational form
    V = FunctionSpace(mesh, ("CG", 1))
    u, v = TrialFunction(V), TestFunction(V)
    F = kappa * inner(grad(u), grad(v)) * dx - inner(f, v) * dx

    # boundaries
    # Dirichlet BCs
    u1 = dolfinx.fem.Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(1)

    x1facet = np.array(ft.indices[ft.values == right_cc_marker])
    x1bc = dolfinx.fem.dirichletbc(u1, dolfinx.fem.locate_dofs_topological(V, 1, x1facet))

    bcs = [x1bc]

    # Neumann boundary - insulated
    F += inner(g, v) * ds(2)

    # Robin boundary - variable area - set kinetics expression
    F += r * inner(u - s, v) * ds(3)

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
    pyvista.set_jupyter_backend("trame")
    pyvista_cells, cell_types, geometry = create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)
    grid.point_data["u"] = uh.x.array
    grid.set_active_scalars("u")


    plotter.add_text("potential", position="lower_edge", font_size=14, color="black")
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()


######################## INTERACTIVITY ####################################

class VizRoutine:
    def __init__(self, c, r, kappa):
        self.kwargs = {
            'c': c,
            'r': r,
            'kappa': kappa,
        }

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def update(self):
        run_model(**self.kwargs)
        return


engine = VizRoutine(c=c, r=r, kappa=0.5)

plotter.add_slider_widget(
    callback=lambda value: engine('kappa', value),
    rng=[0, 10],
    value=0.5,
    title="kappa",
    pointa=(0.025, 0.925),
    pointb=(0.31, 0.925),
    style='modern',
)

plotter.show()