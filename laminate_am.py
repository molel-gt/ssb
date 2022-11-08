#!/usr/bin/env python3

import gmsh
import meshio
import numpy as np

import commons, geometry


markers = commons.SurfaceMarkers()
phases = commons.Phases()


# meshing
resolution = 0.5

W = 20
L = 50
L_coating = L - 10 * resolution
d = 2
corners = list(reversed(np.arange(0.5 * d, W, d)))

points = [
    (0, 0, 0),
    (L, 0, 0),
    (L, W, 0),
    (0, W, 0),
    (0, corners[0], 0),
]
for i, p in enumerate(corners[:-1]):
    points += [
                (L_coating, p, 0),
                (L_coating, corners[i + 1], 0),
                (0, corners[0], 0),
            ]

gmsh.initialize()
gmsh.model.add("Laminate AM")
gmsh.option.setNumber("Mesh.MeshSizeMin", resolution)
gmsh_points = []
for p in points:
    tag = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    gmsh_points.append(tag)
gmsh.model.occ.synchronize()

lines = []
for i in range(-1, len(gmsh_points) - 1):
    lines.append(
        gmsh.model.occ.addLine(gmsh_points[i], gmsh_points[i + 1])
    )

curveloop = gmsh.model.occ.addCurveLoop(lines)
surface = gmsh.model.occ.addPlaneSurface((1, curveloop))
gmsh.model.occ.synchronize()
all_lines = gmsh.model.getEntities(dim=1)
walls = []
left_cc = []
right_cc = []
for line in all_lines:
    com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
    if np.isclose(com[1], 0) or np.isclose(com[1], W):
        walls.append(line[1])
    elif np.isclose(com[0], L):
        right_cc.append(line[1])
    else:
        left_cc.append(line[1])
left_cc_t = gmsh.model.addPhysicalGroup(1, left_cc, markers.left_cc)
gmsh.model.setPhysicalName(1, left_cc_t, "left_cc")
right_cc_t = gmsh.model.addPhysicalGroup(1, right_cc, markers.right_cc)
gmsh.model.setPhysicalName(1, right_cc_t, "right_cc")
insulated_t = gmsh.model.addPhysicalGroup(1, walls, markers.insulated)
gmsh.model.setPhysicalName(1, insulated_t, "insulated")
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("mesh/laminate/mesh.msh")
gmsh.finalize()

mesh_from_file = meshio.read(f"mesh/laminate/mesh.msh")
triangle_mesh = geometry.create_mesh(mesh_from_file, "triangle")
meshio.write("mesh/laminate/tria.xdmf", triangle_mesh)
line_mesh = geometry.create_mesh(mesh_from_file, "line")
meshio.write("mesh/laminate/line.xdmf", line_mesh)