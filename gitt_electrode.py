#!/usr/bin/env python3

import gmsh
import meshio
import numpy as np

import commons, geometry


markers = commons.SurfaceMarkers()
phases = commons.Phases()

N_pieces = 1000
dy = 20 / N_pieces

L = 20
W = 20

# meshing
resolution = 0.05
L_coating = L - 5 * resolution

gmsh.initialize()
gmsh.model.add("AM/SE")
# gmsh.option.setNumber("Mesh.MeshSizeMax", resolution)

points = []
for i in range(0, 1000):
    if i % 2 == 0:
        count = 0
        for p in [
                (L_coating, np.round(i * dy, 2), 0),
                (L_coating, np.round((i + 1) * dy, 2), 0),
                (0, np.round((i + 1) * dy, 2), 0),
                (0, np.round((i + 2) * dy, 2), 0)
                ]:
            tag = gmsh.model.occ.addPoint(*p, meshSize=resolution)
            points.append(tag)
            print(p)

points.append(gmsh.model.occ.addPoint(L_coating, W - dy, 0, meshSize=resolution))
points.append(gmsh.model.occ.addPoint(L_coating, W, 0, meshSize=resolution))
for p in [(L, W, 0), (L, 0, 0)]:
    tag = gmsh.model.occ.addPoint(*p)
    points.append(tag)
    print(p)
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