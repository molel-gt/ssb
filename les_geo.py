#!/usr/bin/env python3
import os
import gmsh

Lx = 200
Ly = 100
points_left = [
    (0, 0, 0),
    (0, 100, 0),
]
resolution = 1
points_right = [
    (Lx, 0, 0),
    (Lx, Ly, 0)
]

R_major = 20
R_minor = 2
center = (0.5 * Lx, 0.5 * Ly, 0)
major_left = (0.5 * Lx - R_major, 0.5*Ly, 0)
major_right = (0.5*Lx + R_major, 0.5*Ly, 0)
minor_up = (0.5*Lx, 0.5*Ly + R_minor, 0)
center_left = (0.5*Lx - 0.5 * R_major, 0.5*Ly, 0)
center_right = (0.5*Lx + 0.5 * R_major, 0.5*Ly, 0)

gmsh.initialize()
gmsh.model.add('disc')
model = gmsh.model.occ
points = []

points.append(model.addPoint(*points_left[0]))
points.append(model.addPoint(*points_left[1]))
points.append(model.addPoint(*points_right[0]))
points.append(model.addPoint(*points_right[1]))

ellipse_points = []
ellipse_points.append(model.addPoint(*major_left))
ellipse_points.append(model.addPoint(*center_left))
ellipse_points.append(model.addPoint(*center))
ellipse_points.append(model.addPoint(*center_right))
ellipse_points.append(model.addPoint(*major_right))
ellipse_points.append(model.addPoint(*minor_up))

lines = []
lines.append(model.addLine(points[0], points[2]))
lines.append(model.addLine(points[2], points[3]))
lines.append(model.addLine(points[3], points[1]))
lines.append(model.addLine(points[1], points[0]))
rect_loop = model.addCurveLoop(lines)

ellipse_arc = []
ellipse_arc.append(model.addEllipseArc(ellipse_points[-1], ellipse_points[2], ellipse_points[1], ellipse_points[0]))
ellipse_arc.append(model.addEllipseArc(ellipse_points[-1], ellipse_points[2], ellipse_points[-3], ellipse_points[-2]))
line_bot = [model.addLine(ellipse_points[0], ellipse_points[-2])]
slice_loop = model.addCurveLoop(ellipse_arc + line_bot)
surf = model.addPlaneSurface([rect_loop] + [slice_loop])
model.synchronize()
gmsh.model.addPhysicalGroup(2, [surf], 1, "domain")
model.synchronize()
gmsh.model.addPhysicalGroup(1, [lines[-1]], 2, "inlet")
gmsh.model.addPhysicalGroup(1, [lines[1]], 3, "outlet")
gmsh.model.addPhysicalGroup(1, [lines[0], lines[-2]], 4, "walls")
gmsh.model.addPhysicalGroup(1, ellipse_arc + line_bot, 5, "arc")
model.synchronize()

gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "CurvesList", ellipse_arc + line_bot)

gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "IField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", resolution/5)
gmsh.model.mesh.field.setNumber(2, "SizeMax", resolution * 5)
gmsh.model.mesh.field.setNumber(2, "DistMin", resolution)
gmsh.model.mesh.field.setNumber(2, "DistMax", 10 * resolution)

gmsh.model.mesh.field.add("Max", 5)
gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
gmsh.model.mesh.field.setAsBackgroundMesh(5)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("mesh.msh")
