#!/usr/bin/env python3
import subprocess
import meshio
import numpy as np

import commons, geometry


markers = commons.SurfaceMarkers()
phases = commons.Phases()

# meshing
resolution = 0.1

msh_fpath = "mesh/laminate/mesh.msh"
res = subprocess.check_call(f"gmsh -3 gitt.geo -o {msh_fpath}", shell=True)

mesh_from_file = meshio.read(f"{msh_fpath}")
triangle_mesh = geometry.create_mesh(mesh_from_file, "triangle")
meshio.write("mesh/laminate/tria.xdmf", triangle_mesh)
line_mesh = geometry.create_mesh(mesh_from_file, "line")
meshio.write("mesh/laminate/line.xdmf", line_mesh)