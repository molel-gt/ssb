#!/usr/bin/env python3

import argparse
import json
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
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="current_dist")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid',  nargs='?', const=1, default='80-20-20')
    parser.add_argument('--resolution', help=f'max resolution resolution', nargs='?', const=1, default=0.1, type=float)
    parser.add_argument("--refine", help="compute current distribution stats", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    micron = 1e-6
    resolution = args.resolution

    Lx, Ly, Lz = [float(val) * micron for val in args.dimensions.split("-")]
    LY = Ly/Lx
    LZ = Lz/Lx
    LX = Lx/Lx
    workdir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], args.name_of_study, args.dimensions, f'{args.resolution}')
    if args.refine:
        workdir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], args.name_of_study, args.dimensions, f'{args.resolution}', 'refined')
    utils.make_dir_if_missing(workdir)
    output_meshfile = os.path.join(workdir, 'mesh.msh')
    output_metafile = os.path.join(workdir, 'geometry.json')
    markers = commons.Markers()

    points_left = [
        (0, 0, 0),
        (0, 0.5*LY, 0)
    ]
    points_right = [
        (LX, 0, 0),
        (LX, 0.5*LY, 0)
    ]
    points_mid = [
        (0.3125 * LX, 0, 0),
        (0.3125 * LX, 0.25 * LY, 0),
        (0.9375 * LX, 0.25 * LY, 0),
        (0.9375 * LX, 0.5*LY, 0)
    ]
    points = []
    lines = []
    gmsh.initialize()
    gmsh.model.add('full-cell')
    if not args.refine:
        gmsh.option.setNumber('Mesh.MeshSizeMax', resolution)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    #      Points Numbering
    # 1-----------------------------2-----------7
    # |                             |           |
    # |                             |           |
    # |                             |           |
    # |             4---------------3           |
    # |             |                           |
    # |             |                           |
    # 0-------------5---------------------------6 
    # geometry is revolved around axis 0-5-6 for angle of 2*pi
    points.extend([gmsh.model.occ.addPoint(*p) for p in points_left])
    points.extend([gmsh.model.occ.addPoint(*p) for p in reversed(points_mid)])
    points.extend([gmsh.model.occ.addPoint(*p) for p in points_right])
    gmsh.model.occ.synchronize()
    for idx in range(0, 5):
        lines.append(
            gmsh.model.occ.addLine(points[idx], points[idx+1])
        )
    lines.append(gmsh.model.occ.addLine(points[5], points[0]))
    lines.append(gmsh.model.occ.addLine(points[5], points[6]))
    lines.append(gmsh.model.occ.addLine(points[6], points[7]))
    lines.append(gmsh.model.occ.addLine(points[7], points[2]))

    gmsh.model.occ.synchronize()
    se_loop = gmsh.model.occ.addCurveLoop([lines[idx] for idx in range(6)])
    pe_loop = gmsh.model.occ.addCurveLoop([lines[idx] for idx in [2, 3, 4, 6, 7, 8]])
    gmsh.model.occ.synchronize()
    se_phase = gmsh.model.occ.addPlaneSurface([se_loop])
    pe_phase = gmsh.model.occ.addPlaneSurface([pe_loop])
    gmsh.model.occ.synchronize()
    old_surfs = gmsh.model.getEntities(2)
    objs = gmsh.model.occ.revolve(old_surfs, 0, 0, 0, -1, 0, 0, 2*np.pi, heights=[], recombine=True)
    gmsh.model.occ.synchronize()
    vols = [tag for tag in objs if tag[0] == 3]

    gmsh.model.addPhysicalGroup(3, [vols[0][1]], markers.electrolyte, "electrolyte")
    gmsh.model.addPhysicalGroup(3, [vols[1][1]], markers.positive_am, "positive_am")
    gmsh.model.occ.synchronize()
    if len(vols) != 2:
        raise ValueError("Expected 2 volumes")

    left = []
    right = []
    interface = []
    insulated_electrolyte = []
    insulated_positive_am = []
    surfs = gmsh.model.getEntities(2)
    surfs.remove(old_surfs[0])
    surfs.remove(old_surfs[1])
    for surf in surfs:
        com = gmsh.model.occ.getCenterOfMass(*surf)
        if np.isclose(com[0], 0):
            left.append(surf[1])
        elif np.isclose(com[0], 0.625):
            # print(surf)
            interface.append(surf[1])
        elif np.isclose(com[0], 1):
            right.append(surf[1])
        else:
            if np.isclose(com[0], 0.46875):
                insulated_electrolyte.append(surf[1])
            elif np.isclose(com[0], 0.96875):
                insulated_positive_am.append(surf[1])
            else:
                if np.isclose(com[1], 0):
                    interface.append(surf[1])
                else:
                    raise ValueError("Unexpected surfaces")
    gmsh.model.addPhysicalGroup(2, left, markers.left, "left")
    gmsh.model.addPhysicalGroup(2, right, markers.right, "right")
    gmsh.model.addPhysicalGroup(2, interface, markers.electrolyte_v_positive_am, "electrolyte_v_positive_am")
    gmsh.model.addPhysicalGroup(2, insulated_electrolyte, markers.insulated_electrolyte, "insulated_electrolyte")
    gmsh.model.addPhysicalGroup(2, insulated_positive_am, markers.insulated_positive_am, "insulated_positive_am")
    gmsh.model.occ.synchronize()

    if args.refine:
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "FacesList", left + interface + right)
        
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", resolution/10)
        gmsh.model.mesh.field.setNumber(2, "LcMax", resolution)
        gmsh.model.mesh.field.setNumber(2, "DistMin", resolution)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 5 * resolution)
        
        gmsh.model.mesh.field.add("Max", 5)
        gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(5)
        gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(3)
    gmsh.write(output_meshfile)
    gmsh.finalize()
