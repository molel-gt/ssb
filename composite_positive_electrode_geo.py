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
    parser.add_argument("--dimensions", help="integer unscaled of LX-LY-LZ of the grid", nargs='?', const=1, default='470-470-45')
    parser.add_argument('--lcat', help=f'integer unscaled +ve electrode thickness', type=int, nargs='?', const=1, default=30)
    parser.add_argument('--lsep', help=f'integer unscaled of separator thickness', type=int, nargs='?', const=1, default=15)
    parser.add_argument('--radius', help=f'integer unscaled +AM particle radius', nargs='?', const=1, default=6, type=int)
    parser.add_argument('--eps_am', help=f'positive active material volume fraction', type=float, required=True)
    parser.add_argument('--se_pos_am_area_frac', help=f'se/+ve am contact fraction', nargs='?', const=1, default=1, type=float)
    parser.add_argument('--resolution', help=f'max resolution resolution (microns)', nargs='?', const=1, default=1, type=float)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?', const=1, default="CONTACT_LOSS_SCALING")
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="reaction_distribution")
    parser.add_argument("--refine", help="compute current distribution stats", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    start_time = timeit.default_timer()
    Lx, Ly, Lz = [int(v) for v in args.dimensions.split("-")]
    if Lz != (args.lsep + args.lcat):
        raise ValueError("Cannot resolve dimensions, please check lsep and lcat")

    # if not np.isclose(args.se_pos_am_area_frac, 1):
    #     raise ValueError("Does not handle contact loss between SE and Positive AM")
    scaling = configs.get_configs()[args.scaling]
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])
    LX = Lx * scale_x
    LY = Ly * scale_y

    Rp = args.radius * scale_x
    Lsep = args.lsep * scale_x
    Lcat = args.lcat * scale_x
    LZ = (args.lcat + args.lsep) * scale_x

    df = scale_x * 470 * pd.read_csv(f'centers/{args.eps_am}.csv')
    if args.refine:
        outdir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], args.name_of_study, args.dimensions, f'{args.lsep}-{args.lcat}', str(args.img_id), str(args.eps_am), str(args.se_pos_am_area_frac), str(args.resolution))
    else:
        outdir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], args.name_of_study, args.dimensions, f'{args.lsep}-{args.lcat}', str(args.img_id), str(args.eps_am), str(args.se_pos_am_area_frac), "unrefined", str(args.resolution))
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
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 1)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 20)
    gmsh.option.setNumber('Mesh.AngleToleranceFacetOverlap', 0.075)
    gmsh.option.setNumber("General.Verbosity", 3)
    if not args.refine:
        gmsh.option.setNumber('Mesh.CharacteristicLengthMax', resolution)
    z0_points = [
        (0, 0, 0),
        (LX, 0, 0),
        (LX, LY, 0),
        (0, LY, 0),
    ]
    zL_points = [
        (0, 0, LZ),
        (LX, 0, LZ),
        (LX, LY, LZ),
        (0, LY, LZ),
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

    # gmsh.model.occ.synchronize()
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

        hull_lines = []
        for i in range(-1, len(hull_points) - 1):
            idx = gmsh.model.occ.addLine(hull_points[i], hull_points[i + 1])
            hull_lines.append(idx)

        idx = gmsh.model.occ.addCurveLoop(hull_lines)
        side_loops.append(idx)
        idx2 = gmsh.model.occ.addPlaneSurface((idx, ))
        left.append(idx2)

    middle = [gmsh.model.occ.addPlaneSurface((loops[1], ))]

    for vv in loops[2:]:
        idx = gmsh.model.occ.addPlaneSurface((vv, ))
        insulated.append(idx)

    insulated += [gmsh.model.occ.addPlaneSurface((loops[0], *side_loops))]

    gmsh.model.occ.healShapes()
    surfaces_1 = tuple(left + insulated + middle)
    box_se = gmsh.model.occ.getEntities(3)[0][1]
    box_am = gmsh.model.occ.addBox(0, 0, LZ - Rp, LX, LY, Rp)

    cylinders = []
    spheres = []

    centers = []

    for idx in range(df.shape[0]):
        x, y = df.loc[idx, :]
        if (x + Rp) >= LX or (y + Rp) >= LY:
            continue
        if (x - Rp) <= 0 or (y - Rp) <= 0:
            continue
        centers.append((x, y, LZ - Rp))
        cyl = gmsh.model.occ.addCylinder(x, y, Lsep, 0, 0, Lcat - Rp, Rp)
        cylinders.append((3, cyl))

    merged = gmsh.model.occ.fuse([(3, box_am)], cylinders)
    gmsh.model.occ.cut([(3, box_se)], merged[0], removeTool=False)
    gmsh.model.occ.synchronize()
    pieces = np.linspace(args.lsep, Lz - args.radius, num=11)
    circle_pos = []
    h = scale_z * (pieces[1]  - pieces[0])
    se_am_contact = []
    se_am_no_contact = []
    for idx, piece in enumerate(pieces[:-1]):
        next_mid = piece * scale_z + (1 - args.se_pos_am_area_frac) * h
        circle_pos.append(next_mid)
        se_am_no_contact.append(0.5 * (scale_z * piece + next_mid))
        next_mid2 = pieces[idx+1] * scale_z
        se_am_contact.append(0.5 * (next_mid + next_mid2))
        if idx != 0:
            circle_pos.append(piece * scale_z)

    vols = gmsh.model.occ.getEntities(3)
    circles = []
    for center in centers:
        x, y, _ = center
        for pos in circle_pos:
            circles.append((1, gmsh.model.occ.addCircle(x, y, pos, Rp)))
    ov, ovv = gmsh.model.occ.fragment(vols, circles)
    gmsh.model.occ.synchronize()
    vols = gmsh.model.occ.getEntities(3)

    gmsh.model.addPhysicalGroup(3, [vols[0][1]], markers.electrolyte, "electrolyte")
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [vols[1][1]], markers.positive_am, "positive_am")
    gmsh.model.occ.synchronize()
    print("Generating surface tags..")
    left_surfs = []
    def is_on_the_walls(x, y, z):
        y_planes = np.isclose(y, 0) or np.isclose(y, Ly)
        x_planes = np.isclose(x, 0) or np.isclose(x, Lx)
        return x_planes or y_planes

    for surf in gmsh.model.occ.getEntities(2):
        com = gmsh.model.occ.getCenterOfMass(surf[0], surf[1])
        x = com[0] / scale_x
        y = com[1] / scale_y
        z = com[2] / scale_z
        if np.isclose(z, 0, atol=1):
            left_surfs.append(surf[1])
            continue
        elif np.isclose(com[2], LZ):
            right.append(surf[1])
        elif np.isclose(com[2], 0.5*(LZ - Rp)) and is_on_the_walls(x, y, z):
            insulated_se.append(surf[1])
        elif np.isclose(com[2], LZ - 0.5 * Rp) and is_on_the_walls(x, y, z):
            insulated_am.append(surf[1])
        elif np.any(np.isclose(com[2], se_am_no_contact, atol=1e-9)):
            insulated_am.append(surf[1])
        elif np.any(np.isclose(com[2], se_am_contact, atol=1e-9)):
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
    gmsh.model.mesh.removeDuplicateElements()
    # refinement
    if args.refine:
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "FacesList", left_surfs + interface + right + insulated_se + insulated_am)

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", resolution / 10)
        gmsh.model.mesh.field.setNumber(2, "LcMax", resolution)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 1e-6)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 2e-6)

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
        "Time elapsed (s)": int(timeit.default_timer() - start_time),
    }
    with open(geometry_metafile, "w", encoding='utf-8') as f:
        json.dump(geometry_metadata, f, ensure_ascii=False, indent=4)
    print(f"Wrote {mshpath}")
    print(f"Time elapsed                                    : {int(timeit.default_timer() - start_time):3.5f}s")
