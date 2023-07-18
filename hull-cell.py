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
import vtk  # do not remove, required for latex rendering in PyVista

from dolfinx import fem, mesh, plot
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import (Measure, TestFunction, TrialFunction, 
                 dx, grad, inner, lhs, rhs)


# pyvista.set_jupyter_backend("pythreejs")
plotter = pyvista.Plotter(shape=(1, 2))
comm = MPI.COMM_WORLD

resolution = 0.05e-1

# Channel parameters
L1 = 3e-1
L2 = 2e-1
L = L1 + L2
W = 5e-1

insulated_marker = 2
left_cc_marker = 3
right_cc_marker = 4

# starting
c = [2.5e-1, 2.5e-1, 0]
r = 0.5e-1

# parameters
R = 8.314  # J/K/mol
T = 298  # K
faraday_const = 96485  # C/mol
KAPPA = 0.1  # S/m
alpha_a = 0.5
alpha_c = 0.5


def create_mesh(domain, cell_type, prune_z=False):
    cells = domain.get_cells_type(cell_type)
    cell_data = domain.get_cell_data("gmsh:physical", cell_type)
    points = domain.points[:, :2] if prune_z else domain.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data]})
    return out_mesh


def create_geometry(c, r):
    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', 1)
    gmsh.model.add("hull cell")
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.005)

    circle_tag = gmsh.model.occ.addCircle(*c, r)
    circle_loop = gmsh.model.occ.addCurveLoop([circle_tag])

    points = [gmsh.model.occ.addPoint(*(0, 0, 0)),
              gmsh.model.occ.addPoint(*(0, W, 0)),
              gmsh.model.occ.addPoint(*(L1, W, 0)),
              gmsh.model.occ.addPoint(*(L, 0, 0))]
    
    channel_lines = [gmsh.model.occ.addLine(points[i], points[i+1])
                     for i in range(-1, len(points)-1)]
    channel_loop = gmsh.model.occ.addCurveLoop(channel_lines)
    plane_surface = gmsh.model.occ.addPlaneSurface((1, channel_loop, circle_loop))

    gmsh.model.occ.synchronize()
    surf = gmsh.model.addPhysicalGroup(2, [plane_surface])
    gmsh.model.setPhysicalName(2, surf, "domain")
    ins_tag = gmsh.model.addPhysicalGroup(1, [channel_lines[0], channel_lines[2], circle_tag], insulated_marker)
    gmsh.model.setPhysicalName(1, ins_tag, "insulated")
    left_tag = gmsh.model.addPhysicalGroup(1, [channel_lines[1]], left_cc_marker)
    gmsh.model.setPhysicalName(1, left_tag, "left")
    right_tag = gmsh.model.addPhysicalGroup(1, [channel_lines[3]], right_cc_marker)
    gmsh.model.setPhysicalName(1, right_tag, "right")
    gmsh.model.mesh.generate(dim=2)
    gmsh.write("mesh.msh")
    gmsh.clear()
    gmsh.finalize()

    mesh_from_file = meshio.read("mesh.msh")
    line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
    meshio.write("facet_mesh.xdmf", line_mesh)

    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    meshio.write("mesh.xdmf", triangle_mesh)


def run_model(c=c, r=r, Wa=0.1, W=W, L=L, L2=L2):
    """Wrapper to allow value update on parameter change"""
    new_c = list(c)
    x_join = (L ** 2 / L2 - c[1] + c[0] * L2 / L)/(L2 / L + L / L2)
    y_join = -L / L2 * x_join + L ** 2 / L2

    if c[0] > (L1 - r) and np.linalg.norm([x_join - c[0], y_join - c[1]]) < r:
        new_c[0] = x_join - (r + 5 * resolution) / (2 ** 0.5)
        new_c[1] = y_join - (r + 5 * resolution) / (2 ** 0.5)

    if c[0] <= r:
        new_c[0] = r + 5 * resolution
    new_c[0] = min(new_c[0], L - r / 3 ** 0.5 - 5 * resolution)
    if c[1] <= r:
        new_c[1] = r + 5 * resolution
    if (c[1] + r) > W:
        new_c[1] = W - r - 5 * resolution

    try:
        create_geometry(tuple(new_c), r)
    except:
        create_geometry([2.5e-1, 2.5e-1, 0], r)

    with dolfinx.io.XDMFFile(comm, "mesh.xdmf", "r") as infile2:
        domain = infile2.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)

    with dolfinx.io.XDMFFile(comm, 'facet_mesh.xdmf', "r") as xdmf:
        ft = xdmf.read_meshtags(domain, name="Grid")
    
    ft_tag = mesh.meshtags(domain, domain.topology.dim - 1, ft.indices, ft.values)

    # Define physical parameters and boundary conditions
    phi_m_n = fem.Constant(domain, ScalarType(0))
    phi_m_p = fem.Constant(domain, ScalarType(1))
    f = fem.Constant(domain, ScalarType(0.0))
    n = ufl.FacetNormal(domain)

    ds = Measure("ds", domain=domain, subdomain_data=ft_tag)

    g = fem.Constant(domain, ScalarType(0.0))
    kappa = fem.Constant(domain, ScalarType(KAPPA))
    i_exch = KAPPA * R * T / (L * faraday_const * Wa * (alpha_a + alpha_c))

    # Define function space and standard part of variational form
    V = fem.FunctionSpace(domain, ("CG", 1))
    u, v = TrialFunction(V), TestFunction(V)
    F = inner(kappa * grad(u), grad(v)) * dx - inner(f, v) * dx

    bcs = []

    # Neumann boundary - insulated
    F -= inner(g, v) * ds(insulated_marker)

    # set linear kinetics expression
    F -= inner(i_exch * faraday_const * (1 / R / T) * (phi_m_p - u), v) * ds(left_cc_marker)
    F -= inner(i_exch * faraday_const * (1 / R / T) * (phi_m_n - u), v) * ds(right_cc_marker)

    # Solve linear variational problem
    options = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "ksp_rtol": 1.0e-12,
            }
    a_form = lhs(F)
    L_form = rhs(F)
    problem = fem.petsc.LinearProblem(a_form, L_form, bcs=bcs, petsc_options=options)
    uh = problem.solve()

    # save to file
    with dolfinx.io.XDMFFile(comm, "secondary-potential.xdmf", "w") as outfile:
        outfile.write_mesh(domain)
        outfile.write_function(uh)

    grad_u = ufl.grad(uh)

    W = fem.VectorFunctionSpace(domain, ("Lagrange", 1))
    current_expr = fem.Expression(-kappa * grad_u, W.element.interpolation_points())
    current_h = fem.Function(W)
    current_h.interpolate(current_expr)

    with dolfinx.io.XDMFFile(comm, "secondary-current.xdmf", "w") as file:
        file.write_mesh(domain)
        file.write_function(current_h)

    I_left_cc = fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(left_cc_marker)))
    I_right_cc = fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(right_cc_marker)))
    I_insulated = fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(insulated_marker)))
    print(f"left current [A/m^2]: {I_left_cc:.2e}, right current [A/m^2]: {I_right_cc:.2e}, insulated current [A/m^2]: {I_insulated:.2e}")
    # Visualize solution
    pyvista_cells, cell_types, geometry = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)
    grid.point_data["$\phi$ [V]"] = uh.x.array
    grid.set_active_scalars("$\phi$ [V]")

    plotter.subplot(0, 0)
    plotter.add_title('Potential')
    plotter.add_mesh(grid, pickable=True, opacity=1, name='mesh')
    contours = grid.contour(20, compute_normals=True)
    plotter.add_mesh(contours, color="white", line_width=1, name='contours')
    plotter.view_xy()

    plotter.subplot(0, 1)
    plotter.add_title('Current Density')
    grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)
    vectors = current_h.x.array.real.reshape(-1, 2)
    vectors = np.hstack((vectors, np.zeros((vectors.shape[0], 1))))

    grid.point_data.set_vectors(vectors, 'i [A/m$^{2}$]')
    warped = grid.warp_by_scalar()
    # plotter.add_mesh(warped)
    grid.set_active_vectors("i [A/m$^{2}$]")
    glyphs = grid.glyph(orient="i [A/m$^{2}$]", factor=0.5, tolerance=0.05)
    plotter.add_mesh(glyphs, name='i [A/m$^{2}$]', color='white')
    plotter.add_mesh(grid, pickable=False, opacity=0.5, name='mesh')
    plotter.view_xy()
    plotter.screenshot('figures/hull-cell-demo.png')


class VizRoutine:
    def __init__(self, c, r, Wa):
        self.kwargs = {
            'c': c,
            'r': r,
            'Wa': Wa,
        }

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def update(self):
        with pyvista.VtkErrorCatcher() as error_catcher:
            run_model(**self.kwargs)
        return


if __name__ == '__main__':
    engine = VizRoutine(c=c, r=r, Wa=1)
    plotter.enable_point_picking(pickable_window=False,left_clicking=True, callback=lambda value: engine('c', value.tolist()))
    plotter.add_slider_widget(
        callback=lambda value: engine('Wa', value),
        rng=[0.001, 10],
        value=1,
        title="Wagner Number",
        pointa=(0.6, 0.825),
        pointb=(0.9, 0.825),
        style='modern',
    )
    plotter.show(screenshot='figures/hull-cell-demo.png')
