#!/usr/bin/env python3
# coding: utf-8
import gmsh
import meshio
import numpy as np
import pygmsh


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh


if __name__ == '__main__':
    geometry = pygmsh.geo.Geometry()
    model = geometry.__enter__()

    resolution = 0.1
    Lx = 5
    Ly = 10

    point0 = (0, 0, 0)
    point1 = (Lx, 0, 0)
    point2 = (Lx, Ly, 0)
    point3 = (0, Ly, 0)

    # model.add_point(point0, mesh_size=resolution),
    # model.add_point(point1, mesh_size=resolution),

    points = []
    sine_curve_x = [np.around(v, 1) for v in reversed(np.linspace(0, Lx, int((Lx / resolution) + 1)))]
    sine_curve_y = list(0 + np.sin(2 * np.pi * np.array(sine_curve_x) / (2 * Lx)))
    sine_curve_y = [1.0 if (1.0 - v) < 0.01 else v for v in sine_curve_y]
    zeros = np.zeros(len(sine_curve_x))
    curve_positions = list(zip(sine_curve_x, sine_curve_y, zeros))
    for p in curve_positions:
        points.append(
            model.add_point(p, mesh_size=resolution)
        )
    channel_lines = [model.add_line(points[i], points[i+1]) for i in range(-1, len(points)-1)]
    channel_loop = model.add_curve_loop(channel_lines)

    plane_surface = model.add_plane_surface(channel_loop)
    model.synchronize()

    volume_marker = 6
    model.add_physical([plane_surface], "Volume")
    model.add_physical([channel_lines[0]], "Left")
    model.add_physical([channel_lines[2]], "Right")
    model.add_physical([channel_lines[1], channel_lines[3]], "Insulated")


    geometry.generate_mesh(dim=2)
    gmsh.write("mesh.msh")
    gmsh.clear()
    geometry.__exit__()

    mesh_from_file = meshio.read("mesh.msh")



    line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
    meshio.write("facet_mesh.xdmf", line_mesh)

    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    meshio.write("mesh.xdmf", triangle_mesh)