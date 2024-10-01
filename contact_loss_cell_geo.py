#!/usr/bin/env

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
    if np.isclose(args.se_pos_am_area_frac, 1):
        raise ValueError("Not implemented for no contact loss")
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
    # gmsh.option.setNumber("Mesh.Algorithm", 5)
    # gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    # gmsh.option.setNumber("Mesh.MaxNumThreads2D", 12)
    # gmsh.option.setNumber("Mesh.MaxNumThreads3D", 12)
    if not args.refine:
        gmsh.option.setNumber('Mesh.CharacteristicLengthMax', resolution)
    box = gmsh.model.occ.addBox(0, 0, 0, LX, LY, LZ)

    _, box_surfs = gmsh.model.occ.getSurfaceLoops(box)
    for surf in box_surfs[0]:
        com = gmsh.model.occ.getCenterOfMass(2, surf)
        if np.isclose(com[2], 0):
            main_left_surf = surf

    points_mid = [
    (0, 0, LZ - Rp),
    (LX, 0, LZ - Rp),
    (LX, LY, LZ - Rp),
    (0, LY, LZ - Rp)
    ]
    mid_points = [gmsh.model.occ.addPoint(*p) for p in points_mid]
    middle_lines = [gmsh.model.occ.addLine(mid_points[i], mid_points[i+1]) for i in range(-1, 3)]
    middle_loop = gmsh.model.occ.addCurveLoop(middle_lines)

    circles = []
    circular_loops = []
    disks = []
    curved_surfs = []
    centers = []
    for idx in range(df.shape[0]):
        x, y = df.loc[idx, :]
        if (x + Rp) >= LX or (y + Rp) >= LY:
            continue
        if (x - Rp) <= 0 or (y - Rp) <= 0:
            continue
        centers.append((x, y))
        circle = gmsh.model.occ.addCircle(x, y, LZ - Rp, Rp)
        cloop = gmsh.model.occ.addCurveLoop([circle])
        circles.append(circle)
        circular_loops.append(cloop)
        extrusion = gmsh.model.occ.extrude([(1, circle)], 0, 0, -(Lcat - Rp), recombine=True)
        curved_surf = [s for s in extrusion if s[0] == 2]
        curved_surfs.extend(curved_surf)
        curved_loops = gmsh.model.occ.getCurveLoops(curved_surf[0][1])

        for c in curved_loops[1][0]:
            com = gmsh.model.occ.getCenterOfMass(1, c)
            if np.isclose(com[2], Lsep, atol=1e-9):
                cloop = gmsh.model.occ.addCurveLoop([c])
                disk = gmsh.model.occ.addPlaneSurface([cloop])
                disks.append((2, disk))
    middle_plane = gmsh.model.occ.addPlaneSurface([middle_loop] + circular_loops)

    # Contact Loss at SE/Negative Electrode
    img = np.asarray(plt.imread(f'data/current_constriction/test{str(int(args.img_id))}.tif')[:, :, 0], dtype=np.uint8)
    image = img.copy()
    image[0, :] = 0
    image[-1, :] = 0
    image[:, 0] = 0
    image[:, -1] = 0
    active_surfs_com = []
    left_surfs = []
    side_loops = []
    lines = []
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
            lines.append((1, idx))

        idx = gmsh.model.occ.addCurveLoop(hull_lines)
        side_loops.append(idx)
        idx2 = gmsh.model.occ.addPlaneSurface((idx, ))
        com = gmsh.model.occ.getCenterOfMass(2, idx2)
        active_surfs_com.append([com[0], com[1]])
        left_surfs.append((2, idx2))

    ov0, ovv0 = gmsh.model.occ.fragment([(3, box)] + [(2, s) for s in box_surfs[0]], lines)

    ov, ovv = gmsh.model.occ.fragment([(3, box)], [(2, middle_plane)] + disks + curved_surfs)
    vols = [v for v in ov if v[0] == 3]
    surfs = [v for v in ov if v[0] == 2]
    middle_circles = []
    circle_pos = []
    se_am_contact = []
    se_am_no_contact = []
    pieces = np.linspace(args.lsep, Lz - args.radius, num=11)
    h = pieces[1] - pieces[0]
    for piece in pieces[:-1]:
        h0 = (1 - args.se_pos_am_area_frac) * h * scale_z
        h1 = args.se_pos_am_area_frac * h * scale_z
        se_am_no_contact.append(scale_z * piece + 0.5 * h0)
        se_am_contact.append(scale_z * piece + h0 + 0.5 * h1)
        circle_pos.append(piece * scale_z + h0)
        if not np.isclose(piece, pieces[0]):
            circle_pos.append(piece * scale_z)
    for center in centers:
        for zpos in circle_pos:
            middle_circles.append((1, gmsh.model.occ.addCircle(*center, zpos, Rp)))


    ov2, ovv2 = gmsh.model.occ.fragment(vols + surfs, middle_circles)

    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [vols[0][1]], markers.electrolyte, "electrolyte")
    gmsh.model.addPhysicalGroup(3, [vols[1][1]], markers.positive_am, "positive am")

    def is_on_the_walls(x, y, z):
        y_planes = np.isclose(y, 0) or np.isclose(y, Ly)
        x_planes = np.isclose(x, 0) or np.isclose(x, Lx)
        return x_planes or y_planes

    active_left = []
    inactive_left = []
   
    interface = []
    insulated_am = []
    insulated_se = []
    right = []
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
                area = gmsh.model.occ.getMass(2, surf[1])
                if not np.isclose(area, 800e-6**2):
                    inactive_left.append(surf[1])
            continue
        elif np.isclose(com[2], LZ):
            right.append(surf[1])
        elif np.isclose(com[2], 0.5*(LZ - Rp)) and is_on_the_walls(x, y, z):
            insulated_se.append(surf[1])
        elif np.isclose(com[2], LZ - 0.5 * Rp) and is_on_the_walls(x, y, z):
            insulated_am.append(surf[1])
        else:
            if np.isclose(z, args.lsep):
                interface.append(surf[1])
            elif np.isclose(z, Lz - args.radius):
                area = gmsh.model.occ.getMass(*surf)
                if not np.isclose(area, np.pi * Rp ** 2, atol=1e-9):
                    interface.append(surf[1])
            elif np.any(np.isclose(se_am_contact, com[2], atol=1e-9)):
                interface.append(surf[1])
            elif np.any(np.isclose(se_am_no_contact, com[2], atol=1e-9)):
                insulated_am.append(surf[1])
            else:
                pass
    gmsh.model.addPhysicalGroup(2, active_left, markers.left, "left")
    insulated_se.extend(inactive_left)
    gmsh.model.addPhysicalGroup(2, insulated_se, markers.insulated_electrolyte, "insulated_electrolyte")
    gmsh.model.addPhysicalGroup(2, right, markers.right, "right")
    gmsh.model.addPhysicalGroup(2, insulated_am, markers.insulated_positive_am, "insulated_am")
    gmsh.model.addPhysicalGroup(2, interface, markers.electrolyte_v_positive_am, "electrolyte_positive_am_interface")
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
