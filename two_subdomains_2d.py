#!/usr/bin/env python3

import argparse
import os

import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
import ufl
import warnings

import commons, configs, geometry, utils

warnings.simplefilter('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="lmb_planar")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid',  nargs='?', const=1, default='150-40-0')
    parser.add_argument('--particle_radius', help='radius of particle in pixel units', nargs='?', const=1, default=10, type=float)
    parser.add_argument('--well_depth', help='depth of well in pixel units', nargs='?', const=1, default=20, type=float)
    parser.add_argument('--l_pos', help='thickness of positive electrode in pixel units', nargs='?', const=1, default=75, type=float)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='MICRON_TO_METER', type=str)
    parser.add_argument('--resolution', help=f'max resolution resolution', nargs='?', const=1, default=1, type=float)
    parser.add_argument("--refine", help="compute current distribution stats", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    micron = 1e-6
    resolution = args.resolution * micron
    LX, LY, LZ = [float(val) * micron for val in args.dimensions.split("-")]
    step_length = 2 * args.particle_radius * micron
    step_width1 = args.well_depth * micron
    step_width2 = (args.l_pos - 2 * args.particle_radius) * micron

    name_of_study = args.name_of_study
    dimensions = args.dimensions
    dimensions_ii = f'{int(step_width1/micron)}-{int(step_width2/micron)}-{int(step_length/micron)}'
    workdir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], name_of_study, dimensions, dimensions_ii, str(resolution))
    utils.make_dir_if_missing(workdir)
    output_meshfile = os.path.join(workdir, 'mesh.msh')

    markers = commons.Markers()
    points = [
        (0, 0, 0),
        (0.5 * LX, 0, 0),
        (LX, 0, 0),
        (LX, LY, 0),
        (0.5 * LX, LY, 0),
        (0, LY, 0),
    ]
    gpoints = []
    lines = []

    gmsh.initialize()
    gmsh.model.add('full-cell')
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1 * micron)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", resolution)
    for idx, p in enumerate(points):
        gpoints.append(
            gmsh.model.occ.addPoint(*p)
        )
    gmsh.model.occ.synchronize()
    gmsh.model.occ.synchronize()
    for idx in range(0, len(points)-1):
        lines.append(
            gmsh.model.occ.addLine(gpoints[idx], gpoints[idx+1])
        )
    lines.append(
        gmsh.model.occ.addLine(gpoints[-1], gpoints[0])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints[1], gpoints[4])
    )

    gmsh.model.occ.synchronize()
    ltag = gmsh.model.addPhysicalGroup(1, [lines[-2]], markers.left)
    gmsh.model.setPhysicalName(1, ltag, "left")
    rtag = gmsh.model.addPhysicalGroup(1, [lines[2]], markers.right)
    gmsh.model.setPhysicalName(1, rtag, "right")
    evptag = gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [-1]], markers.electrolyte_v_positive_am)
    gmsh.model.setPhysicalName(1, evptag, "electrolyte_v_positive_am")
    ietag = gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [0, 4]], markers.insulated_electrolyte)
    gmsh.model.setPhysicalName(1, ietag, "insulated_electrolyte")
    ipamtag = gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [1, 3]], markers.insulated_positive_am)
    gmsh.model.setPhysicalName(1, ipamtag, "insulated_positive_am")
    gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [0, 1, 3, 4]], markers.insulated)
    δΩ_tag = gmsh.model.addPhysicalGroup(1, lines[:-1], markers.external)
    gmsh.model.setPhysicalName(1, δΩ_tag, "external")
    gmsh.model.occ.synchronize()
    se_loop = gmsh.model.occ.addCurveLoop([lines[idx] for idx in [0, 6, 4, 5]])
    pe_loop = gmsh.model.occ.addCurveLoop([lines[idx] for idx in [1, 2, 3, 6]])
    gmsh.model.occ.synchronize()
    se_phase = gmsh.model.occ.addPlaneSurface([se_loop])
    pe_phase = gmsh.model.occ.addPlaneSurface([pe_loop])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [se_phase], markers.electrolyte)
    gmsh.model.addPhysicalGroup(2, [pe_phase], markers.positive_am)
    gmsh.model.occ.synchronize()

    if args.refine:
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "EdgesList", [lines[idx] for idx in [-1, -2, 0, 1, 2, 3, 4, 5, 6]])
        
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", 0.1 * micron)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 1 * micron)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1 * micron)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 1 * micron)
        
        gmsh.model.mesh.field.add("Max", 5)
        gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(5)
        gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)

    _, eleTags , _ = gmsh.model.mesh.getElements(dim=2)
    q = gmsh.model.mesh.getElementQualities(eleTags[0], "angleShape")
    angles = []
    for vv in zip(eleTags[0], q):
        angles.append(vv[1])
    print(np.average(angles), np.min(angles), np.max(angles), np.std(angles))

    gmsh.write(output_meshfile)
    gmsh.finalize()
