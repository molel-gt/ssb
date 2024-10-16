#!/usr/bin/env python3
import os
import warnings

import gmsh
import numpy as np

import commons, configs, geometry, utils

warnings.simplefilter('ignore')

run_mesh = True
adaptive_refine = True
micron = 1e-6
resolution = 1 * micron

L_sep = 25 * micron
L_neg_cc = 20 * micron
L_sep_neg_cc = 15 * micron
feature_radius = 5 * micron
disk_radius = 100 * micron
L_total = L_sep + L_neg_cc

name_of_study = "lithium_metal_3d_cc_2d"
dimensions = '75-40-0'
workdir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], name_of_study, dimensions, str(resolution))
utils.make_dir_if_missing(workdir)
output_meshfile = os.path.join(workdir, 'mesh.msh')

markers = commons.Markers()

points = [
    (0, 0, 0),
    # (5 * micron, 0, 0),
    (20 * micron, 0, 0),
    (75 * micron, 0, 0),
    (75 * micron, 40 * micron, 0),
    (20 * micron, 40 * micron, 0),
    (0, 40 * micron, 0),
    # middle
    (20 * micron, 30 * micron, 0),
    (10 * micron, 30 * micron, 0),
    (10 * micron, 10 * micron, 0),
    (20 * micron, 10 * micron, 0),
]
gpoints = []
lines = []

gmsh.initialize()
gmsh.model.add('lithium-metal')
# gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.25*micron)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5 * micron)
gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 1)
gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)

for idx, p in enumerate(points):
    if idx in (0, 5):
        gpoints.append(np.nan)
        continue
    gpoints.append(
        gmsh.model.occ.addPoint(*p)
    )
gmsh.model.occ.synchronize()

for idx in range(5):
    if idx in (0, 4):
        lines.append(np.nan)
        continue
    lines.append(
        gmsh.model.occ.addLine(gpoints[idx], gpoints[idx+1])
    )
lines.append(np.nan)
# lines.append(
#     gmsh.model.occ.addLine(gpoints[5], gpoints[0])
# )
lines.append(
    gmsh.model.occ.addLine(gpoints[4], gpoints[6])
)
lines.append(
    gmsh.model.occ.addLine(gpoints[6], gpoints[7])
)
lines.append(
    gmsh.model.occ.addLine(gpoints[7], gpoints[8])
)
lines.append(
    gmsh.model.occ.addLine(gpoints[8], gpoints[9])
)
lines.append(
    gmsh.model.occ.addLine(gpoints[9], gpoints[1])
)
insulated = [lines[idx] for idx in [0, 4, 1, 3] if not np.isnan(lines[idx])]
gmsh.model.occ.synchronize()
# gmsh.model.addPhysicalGroup(1, [lines[5]], markers.left)
gmsh.model.addPhysicalGroup(1, [lines[2]], markers.right, "right")
gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in range(6, 11)], markers.negative_cc_v_negative_am, "negative_cc_v_negative_am")
# gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [0, 4]], markers.insulated_negative_cc)
gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [1, 3]], markers.insulated_electrolyte, "insulated_electrolyte")
# gmsh.model.addPhysicalGroup(1, insulated, markers.insulated)
gmsh.model.occ.synchronize()
se_loop = gmsh.model.occ.addCurveLoop([lines[idx] for idx in [1, 2, 3, 6, 7, 8, 9, 10]])
# ncc_loop = gmsh.model.occ.addCurveLoop([lines[idx] for idx in [0, 5, 4, 6, 7, 8, 9, 10]])
gmsh.model.occ.synchronize()
se_phase = gmsh.model.occ.addPlaneSurface([se_loop])
# ncc_phase = gmsh.model.occ.addPlaneSurface([ncc_loop])
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(2, [se_phase], markers.electrolyte)
# gmsh.model.addPhysicalGroup(2, [ncc_phase], markers.negative_cc)
gmsh.model.occ.synchronize()

if adaptive_refine:
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "EdgesList", [lines[idx] for idx in range(11) if not np.isnan(lines[idx])])
    
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", 0.1 * micron)
    gmsh.model.mesh.field.setNumber(2, "LcMax", 0.5 * micron)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1 * micron)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 1 * micron)
    
    gmsh.model.mesh.field.add("Max", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)
    gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write(output_meshfile)
gmsh.finalize()