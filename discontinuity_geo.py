#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import gmsh
import meshio
import numpy as np

import commons, geometry, utils


markers = commons.SurfaceMarkers()
CELL_TYPES = commons.CellTypes()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generation of 2D mesh with discontinuity')
    parser.add_argument('--resolution', help='resolution in the bulk of mesh', default=0.1, type=float)
    parser.add_argument('--outdir', help='output directory', required=True)
    args = parser.parse_args()
    # ϵ = np.finfo(float).eps
    outdir = args.outdir
    utils.make_dir_if_missing(outdir)
    msh_fpath = os.path.join(outdir, 'mesh.msh')
    coords = [
        (0, 0, 0),
        (0, 1, 0),
        (0.5, 1, 0),
        (1, 1, 0),
        (1, 0, 0),
        (0.5, 0, 0),
    ]
    gmsh.initialize()
    # gmsh.option.setNumber('General.Verbosity', 1)
    gmsh.model.add("discontinuity")
    # gmsh.option.setNumber("General.ExpertMode", 1)
    gmsh.option.setNumber("Mesh.MeshSizeMin", args.resolution)
    gmsh.option.setNumber("Mesh.MeshSizeMax", args.resolution)
    points = []
    for p in coords:
        points.append(
            gmsh.model.occ.addPoint(*p)
        )
    gmsh.model.occ.synchronize()
    lines = []

    lines.append(gmsh.model.occ.addLine(1, 2))
    lines.append(gmsh.model.occ.addLine(2, 3))
    lines.append(gmsh.model.occ.addLine(3, 4))
    lines.append(gmsh.model.occ.addLine(4, 5))

    lines.append(gmsh.model.occ.addLine(5, 6))
    lines.append(gmsh.model.occ.addLine(6, 1))
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(1, [2, 3], markers.left_cc)
    gmsh.model.addPhysicalGroup(1, [6], markers.right_cc)
    gmsh.model.occ.synchronize()
    loops = []

    loops.append(gmsh.model.occ.addCurveLoop(lines))
    gmsh.model.occ.synchronize()
    surfaces = []
    surfaces.append(gmsh.model.occ.addPlaneSurface(loops))
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, surfaces, 1)
    gmsh.model.occ.synchronize()
    # refinement
    # gmsh.model.mesh.field.add("Distance", 1)
    # gmsh.model.mesh.field.setNumbers(1, "EdgesList", [2, 3, 6])
    # gmsh.model.mesh.field.add("Threshold", 2)
    # gmsh.model.mesh.field.setNumber(2, "IField", 1)
    # gmsh.model.mesh.field.setNumber(2, "LcMin", args.resolution / 100)
    # gmsh.model.mesh.field.setNumber(2, "LcMax", 10 * args.resolution)
    # gmsh.model.mesh.field.setNumber(2, "DistMin", args.resolution)
    # gmsh.model.mesh.field.setNumber(2, "DistMax", 10 * args.resolution)
    # gmsh.model.mesh.field.add("Max", 5)
    # gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
    # gmsh.model.mesh.field.setAsBackgroundMesh(5)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.write(msh_fpath)
    gmsh.finalize()

    msh = meshio.read(f"{msh_fpath}")
    tria_xdmf_unscaled = geometry.create_mesh(msh, CELL_TYPES.triangle)
    tria_xdmf_unscaled.write(os.path.join(outdir, "tria.xdmf"))
    # tria_xdmf_scaled = geometry.scale_mesh(tria_xdmf_unscaled, CELL_TYPES.triangle, scale_factor=scale_factor)
    # tria_xdmf_scaled.write(os.path.join(outdir, "tria.xdmf"))
    line_xdmf_unscaled = geometry.create_mesh(msh, "line")
    line_xdmf_unscaled.write(os.path.join(outdir, "line.xdmf"))
    # line_xdmf_scaled = geometry.scale_mesh(line_xdmf_unscaled, CELL_TYPES.line, scale_factor=scale_factor)
    # line_xdmf_scaled.write(os.path.join(outdir, "line.xdmf"))
