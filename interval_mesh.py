#!/usr/bin/env python3

import os

import gmsh
import commons, utils
N = 100
scale = 1  # 100e-6
points = [(scale*idx/N, 0, 0) for idx in range(int(N+1))]
resolution = 0.01

markers = commons.Markers()

gmsh.initialize()
gmsh.model.add('interval')
gmsh.option.setNumber('Mesh.CharacteristicLengthMax', resolution)
gpoints = [gmsh.model.occ.addPoint(*p) for p in points]

lines = [gmsh.model.occ.addLine(gpoints[int(idx2)], gpoints[int(idx2+1)]) for idx2 in range(N)]

gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(1, lines[:int(N/2)], markers.electrolyte, "electrolyte")
gmsh.model.addPhysicalGroup(1, lines[int(N/2):], markers.positive_am, "positive_am")
gmsh.model.addPhysicalGroup(0, [gpoints[0]], markers.left, "left")
gmsh.model.addPhysicalGroup(0, [gpoints[int(N/2)]], markers.electrolyte_v_positive_am, "interface")
gmsh.model.addPhysicalGroup(0, [gpoints[-1]], markers.right, "right")
gmsh.model.mesh.generate(1)
workdir = "output/secondary_current/100-0-0/"
utils.make_dir_if_missing(workdir)
gmsh.write(os.path.join(workdir, "mesh.msh"))
