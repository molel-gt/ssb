#!/usr/bin/env python3
# coding: utf-8
import gmsh
import meshio
import numpy as np

from mpi4py import MPI

import geometry


def make_shape_1(lx, ly):
    """"""
    x_vals = np.array([0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10])
    y_vals = np.array([0, 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1, 0])

    return x_vals, y_vals


def make_shape_2(lx, ly, resolution):
    """"""
    x = np.array([np.around(v, 4) for v in reversed(np.linspace(0, Lx, int((Lx / resolution) + 1)))])
    sine_curve_y = list(0 + np.around(np.abs(np.sin(2 * np.pi * np.array(x) / (0.5 * 2 * Lx))), 4))
    sine_curve_y = [1.0 if (1.0 - v) < 0.01 else v for v in sine_curve_y]

    return x, sine_curve_y


Lx = 1000
Ly = 1
resolution = 0.1
gmsh.initialize()
# gmsh.option.setNumber("Mesh.MeshSizeMin", 0.01)
# gmsh.option.setNumber("Mesh.MeshSizeMax", 0.01)
gmsh.model.add("li-sse")
meshname = 'coverage'
# # lithium phase
# x, sine_curve_y = make_shape_1(Lx, Ly)
# meshname = "serrated"
# x, sine_curve_y = make_shape_2(Lx, Ly, resolution)
# meshname = "sinecurve"
# zeros = np.zeros(len(x))
# curve_positions = list(zip(x, sine_curve_y, zeros))
# curve_positions = [gmsh.model.occ.add_point(*p) for p in curve_positions]
# arc_lines = [gmsh.model.occ.add_line(curve_positions[i], curve_positions[i+1]) for i in range(-1, len(curve_positions) - 1)]
# arc_loop = gmsh.model.occ.add_curve_loop(arc_lines)
# lithium_surf = gmsh.model.occ.add_plane_surface([arc_loop])
# electrolyte phase
p0 = gmsh.model.occ.add_point(0, 0, 0, meshSize=resolution)
p1 = gmsh.model.occ.add_point(0, Ly, 0, meshSize=resolution)
p2 = gmsh.model.occ.add_point(Lx, Ly, 0, meshSize=resolution)
p3 = gmsh.model.occ.add_point(Lx, 0, 0, meshSize=resolution)
electrolyte_points = [p0, p1, p2, p3]
lines = [gmsh.model.occ.add_line(electrolyte_points[i], electrolyte_points[i+1]) for i in range(-1, len(electrolyte_points) - 1)]
line_loop = gmsh.model.occ.add_curve_loop(lines)
electrolyte_surf = gmsh.model.occ.add_plane_surface([line_loop])
# full_domain = gmsh.model.occ.fragment([(2, lithium_surf)], [(2, electrolyte_surf)])

gmsh.model.occ.synchronize()
surfaces = gmsh.model.getEntities(dim=2)
# lithium_surface = gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], 1)
electrolyte_surface = gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], 1)

gmsh.model.mesh.generate(2)
gmsh.write(f"{meshname}.msh")
gmsh.finalize()

mesh_from_file = meshio.read(f"{meshname}.msh")

triangle_mesh = geometry.create_mesh(mesh_from_file, "triangle")
meshio.write(f"{meshname}.xdmf", triangle_mesh)
