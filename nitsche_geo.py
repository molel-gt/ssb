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

points_all = [
    (0, 0, 0),
    (0, 50, 0),
    (50, 50, 0),
    (50, 0, 0),
    (60, 0, 0),
    (60, 50, 0),
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    # parser.add_argument('--grid_extents', help='Nx-Ny-Nz_Ox-Oy-Oz size_location', required=True)
    parser.add_argument('--root_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='VOXEL_SCALING4', type=str)
    parser.add_argument('--resolution', help='resolution used in gmsh', nargs='?',
                        const=1, default=0.1, type=float)
    args = parser.parse_args()
    utils.make_dir_if_missing(args.root_folder)
    output_meshfile = os.path.join(args.root_folder, 'mesh.msh')
    gmsh.initialize()
    gmsh.model.add('diffusivity')
    gmsh.option.setNumber('Mesh.Smoothing', 100)
    gpoints = []

    for p in points_all:
        gpoints.append(
            gmsh.model.occ.addPoint(*p)
        )

    gmsh.model.occ.synchronize()
    lines = []
    # round loop
    lines.append(
        gmsh.model.occ.addLine(gpoints[0], gpoints[1])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints[1], gpoints[2])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints[2], gpoints[3])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints[3], gpoints[0])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints[3], gpoints[4])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints[4], gpoints[5])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints[5], gpoints[2])
    )

    gmsh.model.occ.synchronize()
    left = [lines[0]]
    right = [lines[5]]
    middle = [lines[2]]
    insulated = [lines[idx] for idx in [1, 6, 3, 4]]
    left_cc = gmsh.model.addPhysicalGroup(1, left, markers.left_cc)
    right_cc = gmsh.model.addPhysicalGroup(1, right, markers.right_cc)
    insulated_ = gmsh.model.addPhysicalGroup(1, insulated, markers.insulated)
    am_se = gmsh.model.addPhysicalGroup(1, middle, markers.am_se_interface)
    gmsh.model.occ.synchronize()
    se_phase = [lines[idx] for idx in [0, 1, 2, 3]]
    am_phase = [lines[idx] for idx in [2, 4, 5, 6]]
    se_loop = gmsh.model.occ.addCurveLoop(se_phase)
    am_loop = gmsh.model.occ.addCurveLoop(am_phase)
    gmsh.model.occ.synchronize()
    se_surf = gmsh.model.occ.addPlaneSurface([se_loop])
    am_surf = gmsh.model.occ.addPlaneSurface([am_loop])
    gmsh.model.occ.synchronize()
    se_domain = gmsh.model.addPhysicalGroup(2, [se_surf], phases.electrolyte)
    am_domain = gmsh.model.addPhysicalGroup(2, [am_surf], phases.active_material)
    gmsh.model.occ.synchronize()

    # refinement
    # resolution = args.resolution
    # gmsh.model.mesh.field.add("Distance", 1)
    # gmsh.model.mesh.field.setNumbers(1, "EdgesList", [insulated_, left_cc, right_cc, am_se])
    #
    # gmsh.model.mesh.field.add("Threshold", 2)
    # gmsh.model.mesh.field.setNumber(2, "IField", 1)
    # gmsh.model.mesh.field.setNumber(2, "LcMin", resolution/100)
    # gmsh.model.mesh.field.setNumber(2, "LcMax", resolution)
    # gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
    # gmsh.model.mesh.field.setNumber(2, "DistMax", 0.5)
    #
    # gmsh.model.mesh.field.add("Max", 5)
    # gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
    # gmsh.model.mesh.field.setAsBackgroundMesh(5)

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
