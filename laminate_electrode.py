#!/usr/bin/env python3

import gmsh
import meshio
import numpy as np

import commons, geometry


markers = commons.SurfaceMarkers()
phases = commons.Phases()

W = 20

# La = 25  # anode
Ls = 25  # separator
Lc = 75  # cathode
L = Ls + Lc
eps_se = 0.3
eps_am = 0.7

# meshing
resolution = 0.05
L_coating = L - 5 * resolution

points = [
    (0, 0, 0),
    (0, W, 0),
    (L_coating, W, 0),
    (L_coating, int(0.75 * W), 0),
    (Ls, int(0.75 * W), 0),
    (Ls, int(0.25 * W), 0),
    (L_coating, int(0.25 * W), 0),
    (L_coating, 0, 0),
    (L, 0, 0),
     (L, W, 0),
]


gmsh.initialize()
gmsh.model.add("AM/SE")
# gmsh.option.setNumber("Mesh.MeshSizeMax", resolution)
gmsh_points = []
for p in points:
    tag = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    gmsh_points.append(tag)

se_lines = []
points_se = gmsh_points[:8]
for i in range(-1, len(points_se) - 1):
    line = gmsh.model.occ.addLine(points_se[i], points_se[i + 1])
    se_lines.append(line)
se_loop = gmsh.model.occ.addCurveLoop(se_lines)
se_channel = gmsh.model.occ.addPlaneSurface((1, se_loop))
am_lines = []
points_am = gmsh_points[2:]
for i in range(-1, len(points_am) - 1):
        line = gmsh.model.occ.addLine(points_am[i], points_am[i + 1])
        am_lines.append(line)
am_loop = gmsh.model.occ.addCurveLoop(am_lines)
am_channel = gmsh.model.occ.addPlaneSurface((2, am_loop))
gmsh.model.occ.fragment([(2, se_channel)], [(2, am_channel)])
gmsh.model.occ.synchronize()
surfaces = gmsh.model.getEntities(dim=2)
lines = gmsh.model.getEntities(dim=1)
s1 = gmsh.model.addPhysicalGroup(2, [se_channel], phases.electrolyte)
gmsh.model.setPhysicalName(2, s1, "SE")
s2 = gmsh.model.addPhysicalGroup(2, [am_channel], phases.active_material)
gmsh.model.setPhysicalName(2, s2, "AM")
gmsh.model.occ.synchronize()
grp1 = gmsh.model.addPhysicalGroup(1, [se_lines[1]], markers.left_cc)
gmsh.model.setPhysicalName(1, grp1, "left_cc")
grp2 = gmsh.model.addPhysicalGroup(1, [am_lines[-1]], markers.right_cc)
gmsh.model.setPhysicalName(1, grp2, "right_cc")
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("mesh/laminate/mesh.msh")
gmsh.finalize()

mesh_from_file = meshio.read(f"mesh/laminate/mesh.msh")
triangle_mesh = geometry.create_mesh(mesh_from_file, "triangle")
meshio.write("mesh/laminate/tria.xdmf", triangle_mesh)
line_mesh = geometry.create_mesh(mesh_from_file, "line")
meshio.write("mesh/laminate/line.xdmf", line_mesh)