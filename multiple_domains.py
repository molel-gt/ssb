#!/usr/bin/env python3
# coding: utf-8
import gmsh
import meshio
import numpy as np

from mpi4py import MPI

import geometry


Lx = 1000
Ly = 1
resolution = 0.1
gmsh.initialize()
gmsh.model.add("li-sse")
meshname = 'coverage'

p0 = gmsh.model.occ.add_point(0, 0, 0, meshSize=resolution)
p1 = gmsh.model.occ.add_point(0, Ly, 0, meshSize=resolution)
p2 = gmsh.model.occ.add_point(Lx, Ly, 0, meshSize=resolution)
p3 = gmsh.model.occ.add_point(Lx, 0, 0, meshSize=resolution)
electrolyte_points = [p0, p1, p2, p3]
lines = [gmsh.model.occ.add_line(electrolyte_points[i], electrolyte_points[i+1]) for i in range(-1, len(electrolyte_points) - 1)]
line_loop = gmsh.model.occ.add_curve_loop(lines)
electrolyte_surf = gmsh.model.occ.add_plane_surface([line_loop])

gmsh.model.occ.synchronize()
surfaces = gmsh.model.getEntities(dim=2)
electrolyte_surface = gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], 1)

gmsh.model.mesh.generate(2)
gmsh.write(f"{meshname}.msh")
gmsh.finalize()

mesh_from_file = meshio.read(f"{meshname}.msh")

triangle_mesh = geometry.create_mesh(mesh_from_file, "triangle")
meshio.write(f"{meshname}.xdmf", triangle_mesh)
