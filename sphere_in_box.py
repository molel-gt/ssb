#!/usr/bin/env python3

import os

import gmsh
import meshio
import numpy as np

import geometry, utils

Lx = Ly = Lz = 50
resolution = 0.1
meshname = 'sphere-in-box'
gmsh.initialize()
gmsh.model.add("constriction")
gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.005)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.25)

channel = (3, gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz))
sphere = (3, gmsh.model.occ.addSphere(25, 25, 25, 20))
gmsh.model.occ.cut([channel], [sphere])
gmsh.model.occ.synchronize()
volumes = gmsh.model.getEntities(dim=3)
for idx, v in enumerate(volumes):
    gmsh.model.addPhysicalGroup(3, [v[1]], idx)

surfaces = gmsh.model.occ.getEntities(dim=2)
wall_marker = 15
left_cc = []
right_cc = []
walls = []
spheres = []

for surface in surfaces:
    com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
    if np.isclose(com[1], 0):
        left_cc.append(surface[1])
    elif np.isclose(com[1], Ly):
        right_cc.append(surface[1])
    elif np.isclose(com[2], 0) or np.isclose(com[2], Lz) or np.isclose(com[0], 0) or np.isclose(com[0], Lx):
        walls.append(surface[1])
    else:
        spheres.append(surface[1])
left_cc = gmsh.model.addPhysicalGroup(2, left_cc)
gmsh.model.setPhysicalName(2, left_cc, "left_cc")
right_cc = gmsh.model.addPhysicalGroup(2, right_cc)
gmsh.model.setPhysicalName(2, right_cc, "right_cc")
insulated = gmsh.model.addPhysicalGroup(2, walls + spheres)
gmsh.model.setPhysicalName(2, insulated, "insulated")

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
gmsh.write(f"{meshname}.msh")
gmsh.finalize()

mesh_from_file = meshio.read(f"{meshname}.msh")
utils.make_dir_if_missing(f"mesh/051-051-051_900-000-000")

tetra_mesh = geometry.create_mesh(mesh_from_file, "tetra")
meshio.write(f"mesh/051-051-051_900-000-000/tetr.xdmf", tetra_mesh)

triangle_mesh = geometry.create_mesh(mesh_from_file, "triangle")
meshio.write(f"mesh/051-051-051_900-000-000/tria.xdmf", triangle_mesh)

