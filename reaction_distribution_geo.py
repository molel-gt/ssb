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


def create_stacked_cylinders(x, y, radius, scale, se_pos_am_area_frac, pieces, model):
    circles_start = []
    circles_mid = []
    _, _, scale_z = scale
    se_am_contact_spot = []
    se_am_no_contact_spot = []
    step_z0 = (1 - se_pos_am_area_frac) * (pieces[1] - pieces[0])
    step_z1 = se_pos_am_area_frac * (pieces[1] - pieces[0])
    for cidx, start_pos in enumerate(pieces[:-1]):
        start_idx = cidx
        end_idx = cidx + 1
        start_l = pieces[start_idx]
        middle_l = pieces[start_idx] + (1 - se_pos_am_area_frac) * (pieces[end_idx] - pieces[start_idx])
        end_l = pieces[end_idx]
        circles_start.append(model.occ.addCircle(x, y, start_l * scale_z, radius))
        circles_mid.append(model.occ.addCircle(x, y, middle_l * scale_z, radius))
        se_am_contact_spot.append(0.5*(end_l + middle_l))
        se_am_no_contact_spot.append(0.5*(start_l + middle_l))
    circle_end = model.occ.addCircle(x, y, pieces[-1] * scale_z, radius)
    circle_loop_end = model.occ.addCurveLoop([circle_end])
    circle_loop_beg = model.occ.addCurveLoop([circles_start[0]])
    # circular planes
    beg_surf = model.occ.addPlaneSurface([circle_loop_beg])
    side_surf_1 = model.occ.extrude([(1, c) for c in circles_start[:1]], 0, 0, scale_z * (pieces[-1] - pieces[0]))
    surfs = [(2, beg_surf)] + [c for c in side_surf_1 if c[0] == 2]

    return surfs, circle_loop_end, circles_start[1:] + circles_mid, se_am_contact_spot, se_am_no_contact_spot


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
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", 12)
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", 12)
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
    active_surfs_com = []
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
        com = gmsh.model.occ.getCenterOfMass(2, idx2)
        active_surfs_com.append([com[0], com[1]])
        left.append(idx2)

    middle = [gmsh.model.occ.addPlaneSurface((loops[1], ))]

    for vv in loops[2:]:
        idx = gmsh.model.occ.addPlaneSurface((vv, ))
        insulated.append(idx)

    gmsh.model.occ.addPlaneSurface((loops[0], *side_loops))

    gmsh.model.occ.healShapes()
    box_se = gmsh.model.occ.getEntities(3)[0][1]

    cylinders = []
    spheres = []
    # for generating contact loss between se and +ve am
    pieces = np.linspace(args.lsep, Lz - args.radius, num=11)
    se_am_contact = []
    se_am_no_contact = []
    print("Generating cylinders..")
    cyl_surfs = []
    circle_loops = []
    all_circles = []
    for idx in range(df.shape[0]):
        x, y = df.loc[idx, :]
        if (x + Rp) >= LX or (y + Rp) >= LY:
            continue
        if (x - Rp) <= 0 or (y - Rp) <= 0:
            continue
        surfs_label, circle_end, circles, se_am_contact_spot, se_am_no_contact_spot = create_stacked_cylinders(x, y, Rp, [scale_x, scale_y, scale_z], args.se_pos_am_area_frac, pieces, gmsh.model)
        se_am_contact.extend(se_am_contact_spot)
        circle_loops.append(circle_end)
        all_circles.extend(circles)
        se_am_no_contact.extend(se_am_no_contact_spot)
        cyl_surfs.extend(surfs_label)

    print("Generating detailed geometry..")
    points_mid = [
    (0, 0, LZ - Rp),
    (LX, 0, LZ - Rp),
    (LX, LY, LZ - Rp),
    (0, LY, LZ - Rp)
    ]
    mid_points = [gmsh.model.occ.addPoint(*p) for p in points_mid]
    middle_lines = [gmsh.model.occ.addLine(mid_points[i], mid_points[i+1]) for i in range(-1, 3)]
    middle_loop = gmsh.model.occ.addCurveLoop(middle_lines)
    middle_plane = gmsh.model.occ.addPlaneSurface([middle_loop] + circle_loops)
    ov1, ovv1 = gmsh.model.occ.fragment([(3, box_se)], cyl_surfs + [(2, middle_plane)] + [(1, c) for c in all_circles])
    gmsh.model.occ.synchronize()
    vols = gmsh.model.occ.getEntities(3)
    gmsh.model.occ.synchronize()
    for v in vols:
        com = gmsh.model.occ.getCenterOfMass(*v)
        if com[2] / scale_z < 0.5 * (args.lsep + Lz - args.radius):
            gmsh.model.addPhysicalGroup(3, [v[1]], markers.electrolyte, "electrolyte")
            gmsh.model.occ.synchronize()
        else:
            gmsh.model.addPhysicalGroup(3, [v[1]], markers.positive_am, "positive_am")
            gmsh.model.occ.synchronize()
    print("Generating surface tags..")

    def is_on_the_walls(x, y, z):
        y_planes = np.isclose(y, 0) or np.isclose(y, Ly)
        x_planes = np.isclose(x, 0) or np.isclose(x, Lx)
        return x_planes or y_planes

    active_left = []
    inactive_left = []
    left_com_arr = np.array(active_surfs_com)
    for surf in gmsh.model.occ.getEntities(2):
        com = gmsh.model.occ.getCenterOfMass(surf[0], surf[1])
        x = com[0] / scale_x
        y = com[1] / scale_y
        z = com[2] / scale_z
        if np.isclose(z, 0, atol=1):
            if np.any(np.all(np.isclose([com[0], com[1]], left_com_arr, atol=1e-9), axis=1)):
                active_left.append(surf[1])
            else:
                inactive_left.append(surf[1])
            continue
        elif np.isclose(com[2], LZ):
            right.append(surf[1])
        elif np.isclose(com[2], 0.5*(LZ - Rp)) and is_on_the_walls(x, y, z):
            insulated_se.append(surf[1])
        elif np.isclose(com[2], LZ - 0.5 * Rp) and is_on_the_walls(x, y, z):
            insulated_am.append(surf[1])
        else:
            if np.isclose(args.se_pos_am_area_frac, 1):
                interface.append(surf[1])
            else:
                if np.isclose(z, args.lsep):
                    interface.append(surf[1])
                elif np.isclose(z, Lz - args.radius):
                    area = gmsh.model.occ.getMass(*surf)
                    if not np.isclose(area, np.pi * Rp ** 2, atol=1e-9):
                        interface.append(surf[1])
                elif np.any(np.isclose(se_am_contact, z)):
                    interface.append(surf[1])
                elif np.any(np.isclose(se_am_no_contact, z)):
                    insulated_am.append(surf[1])
                else:
                    pass

    gmsh.model.addPhysicalGroup(2, active_left, markers.left, "left")
    insulated_se.extend(inactive_left)
    gmsh.model.addPhysicalGroup(2, insulated_se, markers.insulated_electrolyte, "insulated_electrolyte")
    gmsh.model.addPhysicalGroup(2, right, markers.right, "right")
    gmsh.model.addPhysicalGroup(2, insulated_am, markers.insulated_positive_am, "insulated_am")
    gmsh.model.addPhysicalGroup(2, interface, markers.electrolyte_v_positive_am, "electrolyte_positive_am_interface")
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.removeDuplicateElements()
    # refinement
    if args.refine:
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "FacesList", active_left + interface + right + insulated_se + insulated_am)

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", resolution / 10)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", resolution)
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
    print(f"Time elapsed                                    : {int(timeit.default_timer() - start_time):3.5f}s")
