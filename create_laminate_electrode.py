#!/usr/bin/env python3

import gmsh
import meshio
import numpy as np

from collections import OrderedDict

import commons, geometry


markers = commons.SurfaceMarkers()

W = 200
L = 100
La = 25  # anode
Ls = 25  # separator
Lc = 75  # cathode
eps_se = 0.3
eps_am = 0.7

# meshing
resolution = 0.1
L_coating = L - resolution


se_points = [
    (0, 0, 0),  # bottom left
    (0, W, 0),  # top left
]
am_points = [
    (L, W, 0), # top right
]

points1 = np.arange(eps_se, W, 1)
points2 = np.arange(0, W, 1)
serrated = sorted(list(points1) + list(points2) + [W], reverse=True)

gmsh.initialize()
gmsh.model.add("AM/SE")
gmsh.option.setNumber("Mesh.MeshSizeMax", resolution)

serrated_points = []
for i in range(1, len(serrated)):
    p = serrated[i]
    if i % 2 == 1:
        if i == 1:
            line_points = [(L_coating, serrated[i - 1], 0), (L_coating, p, 0), (Ls, p, 0)]
        else:
            line_points = [(L_coating, p, 0), (Ls, p, 0)]
    if i % 2 == 0:
        line_points = [(Ls, p, 0), (L_coating, p, 0)]
    serrated_points += line_points

se_points = se_points + serrated_points[:-1]
am_points = serrated_points + list(reversed(am_points))
points = []
for p in se_points:
    tag = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    points.append(tag)
tag = gmsh.model.occ.addPoint(*serrated_points[-1], meshSize=resolution)
points.append(tag)
tag = gmsh.model.occ.addPoint(*(L, W, 0), meshSize=resolution)
points.append(tag)

se_lines = []
points_se = points[:-2]
for i in range(-1, len(points_se) - 1):
    line = gmsh.model.occ.addLine(points_se[i], points_se[i + 1])
    se_lines.append(line)
se_loop = gmsh.model.occ.addCurveLoop(se_lines)
se_channel = gmsh.model.occ.addPlaneSurface((1, se_loop))
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(2, [se_channel])

am_lines = []
points_am = points[2:]
for i in range(-1, len(points_am) - 1):
        line = gmsh.model.occ.addLine(points_am[i], points_am[i + 1])
        am_lines.append(line)
am_loop = gmsh.model.occ.addCurveLoop(am_lines)
am_channel = gmsh.model.occ.addPlaneSurface((2, am_loop))
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(2, [am_channel])

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("mesh/laminate/mesh.msh")
gmsh.finalize()

mesh_from_file = meshio.read(f"mesh/laminate/mesh.msh")
triangle_mesh = geometry.create_mesh(mesh_from_file, "triangle")
meshio.write("mesh/laminate/tria.xdmf", triangle_mesh)
line_mesh = geometry.create_mesh(mesh_from_file, "line")
meshio.write("mesh/laminate/line.xdmf", line_mesh)