#!/usr/bin/env python3
# coding: utf-8
import argparse
import os

import alphashape
import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
import pandas as pd
import scipy

from dolfinx import cpp, default_scalar_type, fem, io, mesh, nls, plot

import commons, configs, geometry, grapher, utils

markers = commons.Markers()
CELL_TYPES = commons.CellTypes()

LNCC = 20e-6
LNAM = 50e-6
LSSE = 25e-6
LPAM = 50e-6
LPCC = 20e-6


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Current Distribution')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="full_cell")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    parser.add_argument("--resolution", help="maximum resolution", nargs='?', const=1, default=1, type=float)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='MICRON_TO_METER', type=str)

    args = parser.parse_args()
    scaling = configs.get_configs()[args.scaling]
    scale_factor = [float(scaling[k]) for k in ['x', 'y', 'z']]
    LX, LY, LZ = [int(v) for v in args.dimensions.split("-")]


    encoding = io.XDMFFile.Encoding.HDF5
    adaptive_refine = True
    micron = 1e-6
    gmsh_points = []
    thickness = 0
    dummy = [
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1)
    ]
    lines = []
    workdir = f"output/{args.name_of_study}/{args.dimensions}/{args.resolution}"
    utils.make_dir_if_missing(workdir)
    output_meshfile = os.path.join(workdir, 'mesh.msh')
    tetr_meshfile = os.path.join(workdir, "tetr.xdmf")
    tria_meshfile = os.path.join(workdir, "tria.xdmf")
    line_meshfile = os.path.join(workdir, "line.xdmf")
    gmsh.initialize()
    gmsh.model.add('full-cell')
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1 * micron)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.75 * micron)
    for item in [0, LNCC, LNAM, LSSE, LPAM, LPCC]:
        gmsh_points.append(gmsh.model.occ.addPoint(thickness + item, 0, 0))
        # gmsh.model.occ.synchronize()
        gmsh_points.append(gmsh.model.occ.addPoint(thickness + item, LY * micron, 0))
        gmsh.model.occ.synchronize()
        thickness += item
    gmsh.model.occ.synchronize()
    for idx, item in enumerate([0, LNCC, LNAM, LSSE, LPAM, LPCC]):
        lines.append(
                     gmsh.model.occ.addLine(gmsh_points[2 * idx], gmsh_points[2 * idx + 1])
                     )
        gmsh.model.occ.synchronize()
        if idx > 0:
            lines.append(
                     gmsh.model.occ.addLine(gmsh_points[2 * idx], gmsh_points[2 * idx - 2])
                     )
            gmsh.model.occ.synchronize()
            lines.append(
                     gmsh.model.occ.addLine(gmsh_points[2 * idx + 1], gmsh_points[2 * idx - 1])
                     )
            gmsh.model.occ.synchronize()
    loops = []
    loops_def = {
        markers.negative_cc: [1, 3, 2, 4],
        markers.negative_am: [2, 7, 5, 6],
        markers.electrolyte: [5, 10, 8, 9],
        markers.positive_am: [8, 13, 11, 12],
        markers.positive_cc: [11, 16, 14, 15],
    }
    for idx, mark in enumerate([markers.negative_cc, markers.negative_am, markers.electrolyte, markers.positive_am, markers.positive_cc]):
        loop = gmsh.model.occ.addCurveLoop([lines[k - 1] for k in loops_def[mark]])
        loops.append(loop)
        gmsh.model.occ.synchronize()
    gmsh.model.occ.synchronize()
    surfs = []
    marked_domains = []
    for loop in loops:
        surfs.append(
                     gmsh.model.occ.addPlaneSurface([loop])
                     )
        gmsh.model.occ.synchronize()
    gmsh.model.occ.synchronize()
    names = ["negative_cc", "negative_am", "electrolyte", "positive_am", "positive_cc"]
    for idx, mark in enumerate([markers.negative_cc, markers.negative_am, markers.electrolyte, markers.positive_am, markers.positive_cc]):
        marked_domains.append(
                              gmsh.model.addPhysicalGroup(2, [surfs[idx]], mark)
                              )
        gmsh.model.setPhysicalName(2, marked_domains[idx], names[idx])
    gmsh.model.occ.synchronize()
    gmsh.model.occ.dilate(gmsh.model.get_entities(0), 0, 0, 0, micron, micron, 0)
    
    gmsh.model.occ.synchronize()
    left = gmsh.model.addPhysicalGroup(1, [lines[0]], markers.left)
    gmsh.model.setPhysicalName(1, left, "left")
    insulated_neg_cc = gmsh.model.addPhysicalGroup(1, [lines[2], lines[3]], markers.insulated_negative_cc)
    gmsh.model.setPhysicalName(1, insulated_neg_cc, "insulated_negative_cc")
    insulated_neg_am = gmsh.model.addPhysicalGroup(1, [lines[5], lines[6]], markers.insulated_negative_am)
    gmsh.model.setPhysicalName(1, insulated_neg_am, "insulated_negative_am")
    insulated_electrolyte = gmsh.model.addPhysicalGroup(1, [lines[8], lines[9]], markers.insulated_electrolyte)
    gmsh.model.setPhysicalName(1, insulated_electrolyte, "insulated_electrolyte")
    insulated_pos_am = gmsh.model.addPhysicalGroup(1, [lines[11], lines[12]], markers.insulated_positive_am)
    gmsh.model.setPhysicalName(1, insulated_pos_am, "insulated_positive_am")
    insulated_pos_cc = gmsh.model.addPhysicalGroup(1, [lines[14], lines[15]], markers.insulated_positive_cc)
    gmsh.model.setPhysicalName(1, insulated_pos_cc, "insulated_positive_cc")
    # neg_cc_v_neg_am = gmsh.model.addPhysicalGroup(1, [lines[1]], markers.negative_cc_v_negative_am)
    # gmsh.model.setPhysicalName(1, neg_cc_v_neg_am, "negative_cc_v_negative_am")
    neg_am_v_electrolyte = gmsh.model.addPhysicalGroup(1, [lines[4]], markers.negative_am_v_electrolyte)
    gmsh.model.setPhysicalName(1, neg_am_v_electrolyte, "negative_am_v_electrolyte")
    elec_v_pos_am = gmsh.model.addPhysicalGroup(1, [lines[7]], markers.electrolyte_v_positive_am)
    gmsh.model.setPhysicalName(1, elec_v_pos_am, "electrolyte_v_positive_am")
    # pos_am_v_pos_cc = gmsh.model.addPhysicalGroup(1, [lines[10]], markers.positive_am_v_positive_cc)
    # gmsh.model.setPhysicalName(1, insulated_pos_cc, "positive_am_v_positive_cc")
    insulated = gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [2, 3, 5, 6, 8, 9, 11, 12, 14, 15]], markers.insulated)
    gmsh.model.setPhysicalName(1, insulated, "insulated")
    right = gmsh.model.addPhysicalGroup(1, [lines[13]], markers.right)
    gmsh.model.setPhysicalName(1, right, "right")
    gmsh.model.occ.synchronize()

    # adaptive refinement
    if adaptive_refine:
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "EdgesList", [lines[idx] for idx in [6]])
        
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", 0.1 * micron)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 1 * micron)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 1 * micron)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 5 * micron)
        
        gmsh.model.mesh.field.add("Max", 5)
        gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(5)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.write(output_meshfile)
    gmsh.finalize()

    mesh_2d = meshio.read(output_meshfile)
    tria_mesh = geometry.create_mesh(mesh_2d, "triangle")
    meshio.write(tria_meshfile, tria_mesh)
    line_mesh = geometry.create_mesh(mesh_2d, "line")
    meshio.write(line_meshfile, line_mesh)

    # tria_mesh_scaled = geometry.scale_mesh(tria_mesh, "triangle", scale_factor=[1e-6, 1, 1])
    # tria_mesh_scaled.write(tria_meshfile)

    # line_mesh_scaled = geometry.scale_mesh(line_mesh, "line", scale_factor=[1e-6, 1, 1])
    # line_mesh_scaled.write(line_meshfile)