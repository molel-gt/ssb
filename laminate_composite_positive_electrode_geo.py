#!/usr/bin/env python

import json
import os
import subprocess
import sys
import timeit

import alphashape
import argparse
import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
import pandas as pd
import scipy
import warnings

import commons, configs, geometry, grapher, utils
warnings.simplefilter('ignore')


markers = commons.Markers()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument('--img_id', help='contact area image index', required=True, type=int)
    parser.add_argument("--dimensions", help="integer representation of LX-LY-LZ of the grid", required=True)
    parser.add_argument('--lsep', help=f'integer representation of separator thickness', type=int, required=True)
    parser.add_argument('--radius', help=f'integer representation of +AM particle radius', type=int, required=True)
    parser.add_argument('--eps_am', help=f'positive active material volume fraction', type=float, required=True)
    parser.add_argument('--resolution', help=f'max resolution resolution (microns)', nargs='?', const=1, default=1, type=float)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', type=str, required=True)
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="contact_loss_lma")
    args = parser.parse_args()
    start_time = timeit.default_timer()
    LX, LY, LZ = [int(v) for v in args.dimensions.split("-")]
    scaling = configs.get_configs()[args.scaling]
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])
    LX = LX * scale_x
    LY = LY * scale_y
    LZ = LZ * scale_z
    Rp = args.radius * scale_x
    Lsep = args.lsep * scale_x
    Lcat = LZ - Lsep
    df = scale_x * 470 * pd.read_csv('data/laminate.csv')
    outdir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], args.name_of_study, args.dimensions, str(args.img_id), str(args.eps_am), str(args.resolution))
    utils.make_dir_if_missing(outdir)
    mshpath = os.path.join(f"{outdir}", "mesh.msh")
    geometry_metafile = os.path.join(outdir, "geometry.json")
    corner_points = [
        (0, 0, 0),
        (LX, 0, 0),
        (LX, LY, 0),
        (0, LY, 0)
    ]

    resolution = args.resolution * 1e-6

    gmsh.initialize()
    gmsh.model.add('area')
    # gmsh.option.setNumber('Mesh.MeshSizeMin', scale_x * args.resolution)
    # gmsh.option.setNumber('Mesh.MeshSizeMax', scale_x * args.resolution)
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    z0_points = [
        (0, 0, 0),
        (LX, 0, 0),
        (LX, LY, 0),
        (0, LY, 0),
    ]
    zL_points = [
        (0, 0, LZ - Rp),
        (LX, 0, LZ - Rp),
        (LX, LY, LZ - Rp),
        (0, LY, LZ - Rp),
    ]

    points0 = []
    points1 = []

    lines = []

    for i in range(4):
        idx = gmsh.model.occ.addPoint(*z0_points[i])
        points0.append(idx)
    for i in range(4):
        idx = gmsh.model.occ.addPoint(*zL_points[i])
        points1.append(idx)

    gmsh.model.occ.synchronize()
    for i in range(-1, 3):
        idx = gmsh.model.occ.addLine(points0[i], points0[i + 1])
        lines.append(idx)

    for i in range(-1, 3):
        idx = gmsh.model.occ.addLine(points1[i], points1[i + 1])
        lines.append(idx)

    # 1 --> 5
    idx = gmsh.model.occ.addLine(points0[1], points1[1])
    lines.append(idx)

    # 2 --> 6
    idx = gmsh.model.occ.addLine(points0[2], points1[2])
    lines.append(idx)

    # 3 --> 7
    idx = gmsh.model.occ.addLine(points0[3], points1[3])
    lines.append(idx)

    # 0 --> 4
    idx = gmsh.model.occ.addLine(points0[0], points1[0])
    lines.append(idx)

    gmsh.model.occ.synchronize()

    loops = []
    # xy sides
    idx = gmsh.model.occ.addCurveLoop(lines[:4])
    loops.append(idx)
    idx = gmsh.model.occ.addCurveLoop(lines[4:8])
    loops.append(idx)

    # xz sides
    idx = gmsh.model.occ.addCurveLoop([lines[1]] + [lines[8]] + [lines[5]] + [lines[11]])
    loops.append(idx)
    idx = gmsh.model.occ.addCurveLoop([lines[3]] + [lines[9]] + [lines[7]] + [lines[10]])
    loops.append(idx)

    # yz sides
    idx = gmsh.model.occ.addCurveLoop([lines[2]] + [lines[8]] + [lines[6]] + [lines[9]])
    loops.append(idx)
    idx = gmsh.model.occ.addCurveLoop([lines[0]] + [lines[11]] + [lines[4]] + [lines[10]])
    loops.append(idx)

    # gmsh.model.occ.synchronize()

    side_loops = []
    insulated = []
    right = []
    left_active = []
    left = []
    interface = []
    insulated_se = []
    insulated_am = []
    process_count = 0
    img = np.asarray(plt.imread(f'data/current_constriction/test{str(int(args.img_id))}.tif')[:, :, 0], dtype=np.uint8)
    image = img.copy()
    image[0, :] = 0
    image[-1, :] = 0
    image[:, 0] = 0
    image[:, -1] = 0
    boundary_pieces, count, points, points_view = geometry.get_phase_boundary_pieces(image)
    for hull in boundary_pieces:
        hull_arr = np.asarray(hull)
        hull_points = []
        for pp in hull[:-1]:
            idx = gmsh.model.occ.addPoint(int(pp[0]) * scale_x, int(pp[1]) * scale_y, 0)
            hull_points.append(idx)
        gmsh.model.occ.synchronize()
        hull_lines = []
        for i in range(-1, len(hull_points) - 1):
            idx = gmsh.model.occ.addLine(hull_points[i], hull_points[i + 1])
            hull_lines.append(idx)

        gmsh.model.occ.synchronize()
        idx = gmsh.model.occ.addCurveLoop(hull_lines)
        side_loops.append(idx)
        idx2 = gmsh.model.occ.addPlaneSurface((idx, ))
        left.append(idx2)
        gmsh.model.occ.synchronize()

    middle = [gmsh.model.occ.addPlaneSurface((loops[1], ))]
    gmsh.model.occ.synchronize()

    for vv in loops[2:]:
        idx = gmsh.model.occ.addPlaneSurface((vv, ))
        insulated.append(idx)
        gmsh.model.occ.synchronize()

    insulated += [gmsh.model.occ.addPlaneSurface((loops[0], *side_loops))]

    gmsh.model.occ.healShapes()
    gmsh.model.occ.synchronize()
    surfaces_1 = tuple(left + insulated + middle)

    gmsh.model.occ.synchronize()

    sloop1 = gmsh.model.occ.addSurfaceLoop(surfaces_1)
    box_am = gmsh.model.occ.addBox(0, 0, LZ - Rp, LX, LY, Rp)

    gmsh.model.occ.synchronize()
    
    # left_surfs = [vv[1] for vv in gmsh.model.occ.getEntities(2) if vv[1] >= 7]
    # print(left_surfs)
    # insulated_se += [2, 3, 4, 5, 6]
    surfs = gmsh.model.occ.getEntities(2)
    for surf in surfs:
        com = gmsh.model.occ.getCenterOfMass(*surf)
        # print(surf, ": ", com)

    box_se = gmsh.model.occ.getEntities(3)[0][1]
    box_am = gmsh.model.occ.getEntities(3)[1][1]
    cylinders = []
    spheres = []
    counter = 3
    for idx in range(df.shape[0]):
        x, y, _ = df.loc[idx, :]
        if (x + Rp) >= LX or (y + Rp) >= LY:
            continue
        if (x - Rp) <= 0 or (y - Rp) <= 0:
            continue
        gmsh.model.occ.addCylinder(x, y, Lsep, 0, 0, Lcat - Rp, Rp, counter)
        cylinders.append((3, counter))
        counter += 1

    ov, ovv = gmsh.model.occ.fragment([(3, box_am)], cylinders)
    copy =gmsh.model.occ.copy(ov[1:])
    gmsh.model.occ.cut([(3, box_se)], copy)

    gmsh.model.occ.synchronize()
    vols = gmsh.model.occ.getEntities(3)

    se_volumes = []
    am_volumes = []
    for (_, vol) in vols:
        com = gmsh.model.occ.getCenterOfMass(3, vol)
        z = com[2] / scale_x
        if np.isclose(z, 0.5 * (Lsep + Lcat - Rp) / scale_x, atol=1):
            se_volumes.append(vol)
        else:
            am_volumes.append(vol)
    se_vol = gmsh.model.addPhysicalGroup(3, se_volumes, markers.electrolyte, "electrolyte")
    gmsh.model.occ.synchronize()
    am_vol = gmsh.model.addPhysicalGroup(3, am_volumes, markers.positive_am, "positive_am")
    gmsh.model.occ.synchronize()
    print("Generating surface tags..")
    left_surfs = []
    for surf in gmsh.model.occ.getEntities(2):
        com = gmsh.model.occ.getCenterOfMass(surf[0], surf[1])
        x = com[0] / scale_x
        y = com[1] / scale_x
        z = com[2] / scale_x
        if np.isclose(z, 0, atol=1):
            # print(surf[1], ": ", com)
            left_surfs.append(surf[1])
            continue
        elif np.isclose(z, (Lsep + Lcat)/scale_x, atol=1):
            right.append(surf[1])
        elif np.isclose(z, 0.5 * (Lsep + Lcat - Rp) / scale_x, atol=1):
            if np.isclose(x, LX/scale_x, atol=1) or np.isclose(y, LY/scale_x, atol=1) or np.isclose(x, 0, atol=1) or np.isclose(y, 0, atol=1):
                insulated_se.append(surf[1])
            else:
                interface.append(surf[1])
        elif np.isclose(z, (Lsep + Lcat - 0.5 * Rp) / scale_x, atol=1):
            if np.isclose(x, LX/scale_x, atol=1) or np.isclose(y, LY/scale_x, atol=1) or np.isclose(x, 0, atol=1) or np.isclose(y, 0, atol=1):
                insulated_am.append(surf[1])
            else:
                interface.append(surf[1])
        else:
            interface.append(surf[1])
    gmsh.model.addPhysicalGroup(2, left_surfs[1:], markers.left, "left")
    insulated_se.append(left_surfs[0])
    gmsh.model.addPhysicalGroup(2, insulated_se, markers.insulated_electrolyte, "insulated_electrolyte")
    gmsh.model.addPhysicalGroup(2, right, markers.right, "right")
    gmsh.model.addPhysicalGroup(2, insulated_am, markers.insulated_positive_am, "insulated_am")
    gmsh.model.addPhysicalGroup(2, interface, markers.electrolyte_v_positive_am, "electrolyte_positive_am_interface")
    gmsh.model.occ.synchronize()
    # refinement
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "FacesList", left_surfs + interface)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", resolution / 2)
    gmsh.model.mesh.field.setNumber(2, "LcMax", resolution)
    gmsh.model.mesh.field.setNumber(2, "DistMin", resolution)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 2 * resolution)

    gmsh.model.mesh.field.add("Max", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)
    gmsh.model.occ.synchronize()
    print("Generating mesh..")
    gmsh.model.mesh.generate(3)
    gmsh.write(mshpath)
    gmsh.finalize()

    geometry_metadata = {
        "max_resolution": args.resolution,
        "dimensions": args.dimensions,
        "scaling": args.scaling,
    }
    with open(geometry_metafile, "w", encoding='utf-8') as f:
        json.dump(geometry_metadata, f, ensure_ascii=False, indent=4)
    print(f"Time elapsed                                    : {int(timeit.default_timer() - start_time):3.5f}s")
