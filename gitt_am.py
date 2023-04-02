#!/usr/bin/env python3
import subprocess
import meshio
import numpy as np

import commons, geometry


markers = commons.SurfaceMarkers()
phases = commons.Phases()
CELL_TYPES = commons.CellTypes()

# meshing
resolution = 0.1
scale_factor = [1e-6, 1e-6, 0]
program = 'semicircle.geo'  # 'gitt.geo'
msh_fpath = "mesh/gitt/mesh.msh"
res = subprocess.check_call(f"gmsh -3 {program} -o {msh_fpath}", shell=True)

msh = meshio.read(f"{msh_fpath}")
tria_xdmf_unscaled = geometry.create_mesh(msh, CELL_TYPES.triangle)
line_xdmf_unscaled = geometry.create_mesh(msh, "line")
tria_xdmf_unscaled.write("mesh/gitt/tria_unscaled.xdmf")
tria_xdmf_scaled = geometry.scale_mesh(tria_xdmf_unscaled, CELL_TYPES.triangle, scale_factor=scale_factor)
tria_xdmf_scaled.write("mesh/gitt/tria.xdmf")
line_xdmf_unscaled.write("mesh/gitt/line_unscaled.xdmf")
line_xdmf_scaled = geometry.scale_mesh(line_xdmf_unscaled, CELL_TYPES.line, scale_factor=scale_factor)
line_xdmf_scaled.write("mesh/gitt/line.xdmf")