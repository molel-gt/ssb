#!/usr/bin/env python3

import gmsh
import meshio
import numpy as np

import commons, geometry


markers = commons.SurfaceMarkers()
phases = commons.Phases()

W = 20
N_pieces = 1000
dy = 20 / N_pieces

La = 25  # anode
Ls = 25  # separator
Lc = 75  # cathode
L = Ls + Lc
eps_se = 0.3
eps_am = 0.7

# meshing
resolution = 0.05
L_coating = L - 5 * resolution


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
# gmsh.option.setNumber("Mesh.MeshSizeMax", resolution)

points = []
for i in range(0, 1000, 3):
    if i % 2 == 0:
        print(i, i+2)
        tag = gmsh.model.occ.addPoint(L_coating, i * dy, 0, meshSize=resolution)
        points.append(tag)
        tag = gmsh.model.occ.addPoint(L_coating, (i + 1) * dy, 0, meshSize=resolution)
        points.append(tag)
        tag = gmsh.model.occ.addPoint(0, (i + 1) * dy, 0, meshSize=resolution)
        points.append(tag)
        tag = gmsh.model.occ.addPoint(0, (i + 2) * dy, 0, meshSize=resolution)
        points.append(tag)
    else:
        continue

points.append(gmsh.model.occ.addPoint(L_coating, W - dy, 0, meshSize=resolution))
points.append(gmsh.model.occ.addPoint(L_coating, W, 0, meshSize=resolution))
for p in [(L, W, 0), (L, 0, 0)]:
    tag = gmsh.model.occ.addPoint(*p)
    points.append(tag)
am_lines = []

for i in range(-1, len(points) - 1):
        line = gmsh.model.occ.addLine(points[i], points[i + 1])
        am_lines.append(line)

am_loop = gmsh.model.occ.addCurveLoop(am_lines)
am_channel = gmsh.model.occ.addPlaneSurface((2, am_loop))
gmsh.model.occ.synchronize()
surfaces = gmsh.model.getEntities(dim=2)
lines = gmsh.model.getEntities(dim=1)

s2 = gmsh.model.addPhysicalGroup(2, [am_channel], phases.active_material)
gmsh.model.setPhysicalName(2, s2, "AM")
gmsh.model.occ.synchronize()
grp1 = gmsh.model.addPhysicalGroup(1, am_lines[0:-3], markers.left_cc)
gmsh.model.setPhysicalName(1, grp1, "left_cc")
grp2 = gmsh.model.addPhysicalGroup(1, [am_lines[-2]], markers.right_cc)
gmsh.model.setPhysicalName(1, grp2, "right_cc")
grp2 = gmsh.model.addPhysicalGroup(1, [am_lines[-3], am_lines[-1]], markers.insulated)
gmsh.model.setPhysicalName(1, grp2, "insulated")
gmsh.model.occ.synchronize()

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("mesh/laminate/mesh.msh")
gmsh.finalize()

mesh_from_file = meshio.read(f"mesh/laminate/mesh.msh")
triangle_mesh = geometry.create_mesh(mesh_from_file, "triangle")
meshio.write("mesh/laminate/tria.xdmf", triangle_mesh)
line_mesh = geometry.create_mesh(mesh_from_file, "line")
meshio.write("mesh/laminate/line.xdmf", line_mesh)