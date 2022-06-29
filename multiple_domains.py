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
meshname = 'coverage'
gmsh.initialize()
gmsh.model.add("constriction")
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)

channel = gmsh.model.occ.add_rectangle(0, 0, 0, Lx, Ly)
# slit_1 = gmsh.model.occ.add_rectangle(0, 0.75 * Ly, 0, 0.25 * Lx, 0.01 * Ly)
# gmsh.model.occ.cut([(2, channel)], [(2, slit_1)])
gmsh.model.occ.synchronize()
surfaces = gmsh.model.getEntities(dim=2)
electrolyte_surface = gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], 1)

gmsh.model.mesh.generate(2)
gmsh.write(f"{meshname}.msh")
gmsh.finalize()

mesh_from_file = meshio.read(f"{meshname}.msh")

triangle_mesh = geometry.create_mesh(mesh_from_file, "triangle")
meshio.write(f"{meshname}.xdmf", triangle_mesh)
