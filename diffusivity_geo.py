#!/usr/bin/env python3
import csv
import os
import timeit

import argparse
import gmsh
import logging
import meshio
import numpy as np

import commons, configs, constants, geometry
import utils

markers = commons.SurfaceMarkers()
phases = commons.Phases()
cell_types = commons.CellTypes()

points_round = [
    (0, 0, 0),
    (140, 0, 0),
    (150, 0, 0),
    (150, 40, 0),
    (140, 40, 0),
    (0, 40, 0),
]
points_mid = [
    (140, 10, 0),
    (50, 10, 0),
    (50, 30, 0),
    (140, 30, 0),
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    # parser.add_argument('--grid_extents', help='Nx-Ny-Nz_Ox-Oy-Oz size_location', required=True)
    parser.add_argument('--root_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='VOXEL_SCALING4', type=str)
    args = parser.parse_args()
    utils.make_dir_if_missing(args.root_folder)
    output_meshfile = os.path.join(args.root_folder, 'mesh.msh')
    gmsh.initialize()
    gmsh.model.add('diffusivity')
    gpoints_round = []
    gpoints_mid = []
    for p in points_round:
        gpoints_round.append(
            gmsh.model.occ.addPoint(*p)
        )
    for p in points_mid:
        gpoints_mid.append(
            gmsh.model.occ.addPoint(*p)
        )
    gmsh.model.occ.synchronize()
    lines = []
    # round loop
    lines.append(
        gmsh.model.occ.addLine(gpoints_round[0], gpoints_round[1])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints_round[1], gpoints_round[2])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints_round[2], gpoints_round[3])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints_round[3], gpoints_round[4])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints_round[4], gpoints_round[5])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints_round[5], gpoints_round[0])
    )
    # middle lines
    lines.append(
        gmsh.model.occ.addLine(gpoints_round[1], gpoints_mid[0])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints_mid[0], gpoints_mid[1])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints_mid[1], gpoints_mid[2])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints_mid[2], gpoints_mid[3])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints_mid[3], gpoints_round[4])
    )
    gmsh.model.occ.synchronize()
    left = [lines[5]]
    right = [lines[2]]
    middle = lines[6:]
    insulated = lines[:2] + lines[3:5]
    left_cc = gmsh.model.addPhysicalGroup(1, left, markers.left_cc)
    right_cc = gmsh.model.addPhysicalGroup(1, right, markers.right_cc)
    insulated_ = gmsh.model.addPhysicalGroup(1, insulated, markers.insulated)
    am_se = gmsh.model.addPhysicalGroup(1, middle, markers.am_se_interface)
    gmsh.model.occ.synchronize()
    se_phase = [lines[idx] for idx in [0, 6, 7, 8, 9, 10, 4, 5]]
    am_phase = [lines[idx] for idx in [6, 7, 8, 9, 10, 3, 2, 1]]
    se_loop = gmsh.model.occ.addCurveLoop(se_phase)
    am_loop = gmsh.model.occ.addCurveLoop(am_phase)
    gmsh.model.occ.synchronize()
    se_surf = gmsh.model.occ.addPlaneSurface([se_loop])
    am_surf = gmsh.model.occ.addPlaneSurface([am_loop])
    gmsh.model.occ.synchronize()
    se_domain = gmsh.model.addPhysicalGroup(2, [se_surf], phases.electrolyte)
    am_domain = gmsh.model.addPhysicalGroup(2, [am_surf], phases.active_material)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(output_meshfile)
    gmsh.finalize()
    scale_factor = [float(configs.get_configs()[args.scaling][idx]) for idx in ['x', 'y', 'z']]
    mesh_2d = meshio.read(output_meshfile)
    tria_mesh = geometry.create_mesh(mesh_2d, "triangle")
    tria_meshfile = os.path.join(args.root_folder, "tria.xdmf")
    line_meshfile = os.path.join(args.root_folder, "line.xdmf")
    meshio.write(tria_meshfile, tria_mesh)
    tria_mesh_scaled = geometry.scale_mesh(tria_mesh, cell_types.triangle, scale_factor=scale_factor)
    tria_mesh_scaled.write(tria_meshfile)
    line_mesh = geometry.create_mesh(mesh_2d, "line")
    meshio.write(line_meshfile, line_mesh)
    tria_mesh_scaled = geometry.scale_mesh(line_mesh, cell_types.line, scale_factor=scale_factor)
    tria_mesh_scaled.write(line_meshfile)
