#! /usr/bin/env python3

import gc
import os
import pickle
import subprocess
import time

import argparse
import gmsh
import logging
import matplotlib.pyplot as plt
import meshio
import numpy as np

from skimage import io

import commons, configs, filter_voxels, geometry, utils


FORMAT = '%(asctime)s: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__file__)
logger.setLevel(configs.get_configs()['LOGGING']['level'])
output_mesh_file = "mesh/gitt_block/msh.msh"
CELL_TYPES = commons.CellTypes()
markers = commons.SurfaceMarkers()
scale_factor = [250e-6, 250e-6, 0]

points = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (1, 0.75, 0),
        (1, 0.25, 0),
        (1, 0, 0),
    ]
gmsh_points = []
gmsh.initialize()
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1e-4)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1e-2)
gmsh.model.add("BLOCK")
for p in points:
    gmsh_points.append(gmsh.model.occ.addPoint(*p))
lines = []
for i in range(-1, len(points)-1):
    lines.append(
        gmsh.model.occ.addLine(gmsh_points[i], gmsh_points[i+1])
        )
gmsh.model.occ.synchronize()
surf = gmsh.model.occ.addCurveLoop(lines)
surf2d = gmsh.model.occ.addPlaneSurface([surf])
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(2, [surf2d])
gmsh.model.occ.synchronize()

left_cc = gmsh.model.addPhysicalGroup(1, lines[0:3], markers.left_cc)
gmsh.model.setPhysicalName(1, left_cc, "left_cc")

right_cc = gmsh.model.addPhysicalGroup(1, [lines[-2]], markers.right_cc)
gmsh.model.setPhysicalName(1, right_cc, "right_cc")

insulated = gmsh.model.addPhysicalGroup(1, [lines[-1], lines[-3]], markers.insulated)
gmsh.model.setPhysicalName(1, insulated, "insulated")
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
 
gmsh.write(output_mesh_file)
gmsh.finalize()
msh = meshio.read(output_mesh_file)
tria_xdmf_unscaled = geometry.create_mesh(msh, CELL_TYPES.triangle)
line_xdmf_unscaled = geometry.create_mesh(msh, "line")
tria_xdmf_unscaled.write("mesh/gitt_block/tria_unscaled.xdmf")
tria_xdmf_scaled = geometry.scale_mesh(tria_xdmf_unscaled, CELL_TYPES.triangle, scale_factor=scale_factor)
tria_xdmf_scaled.write("mesh/gitt_block/tria.xdmf")
line_xdmf_unscaled.write("mesh/gitt_block/line_unscaled.xdmf")
line_xdmf_scaled = geometry.scale_mesh(line_xdmf_unscaled, CELL_TYPES.line, scale_factor=scale_factor)
line_xdmf_scaled.write("mesh/gitt_block/line.xdmf")
print(f"Wrote tria.xdmf and line.xdmf mesh files to directory: mesh/circles")
    