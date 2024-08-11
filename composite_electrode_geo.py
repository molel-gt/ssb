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


def get_circle_arc_points(center, radius):
    x, y, z = center
    arc_points = [
        (x, y + radius, z),
        (x - radius, y, z),
        (x, y - radius, z),
        (x + radius, y, z),
    ]


def add_multiple_curve_loops(circle_arcs, original_pos, heights, model):
    curved_surfs = []
    arcs = circle_arcs
    pos = original_pos
    for h in heights:
        curved_surf = model.extrude(arcs, 0, 0, h, recombine=False)
        pos += h

        surfs = [c for c in curved_surf if c[0] == 2]
        curved_surfs.extend(surfs)
        lines = [c for c in curved_surf if c[0] == 1]
        circle_arcs_2 = []
        for c in lines:
            com = gmodel.getCenterOfMass(*c)
            if np.isclose(com[2], pos):
                circle_arcs_2.append(c)
        arcs = circle_arcs_2

    return curved_surfs, arcs

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
    points_left = [
        (0, 0, 0),
        (LX, 0, 0),
        (LX, LY, 0),
        (0, LY, 0)
    ]

    points_mid = [
        (0, 0, LZ - Rp),
        (LX, 0, LZ - Rp),
        (LX, LY, LZ - Rp),
        (0, LY, LZ - Rp)
    ]

    points_right = [
        (0, 0, LZ),
        (LX, 0, LZ),
        (LX, LY, LZ),
        (0, LY, LZ)
    ]

    gmsh.initialize()
    gmsh.model.add('geo')
    gmsh.option.setNumber('Mesh.SaveAll', 1)
    gmodel = gmsh.model.occ

    points = [gmodel.addPoint(*p) for p in points_left]
    points.extend([gmodel.addPoint(*p) for p in points_mid])
    points.extend([gmodel.addPoint(*p) for p in points_right])

    # lines left
    lines_left = [gmodel.addLine(points[0], points[1])]
    lines_left.append(gmodel.addLine(points[1], points[2]))
    lines_left.append(gmodel.addLine(points[2], points[3]))
    lines_left.append(gmodel.addLine(points[3], points[0]))

    # lines mid
    lines_mid = [gmodel.addLine(points[4], points[5])]
    lines_mid.append(gmodel.addLine(points[5], points[6]))
    lines_mid.append(gmodel.addLine(points[6], points[7]))
    lines_mid.append(gmodel.addLine(points[7], points[4]))

    # lines right
    lines_right = [gmodel.addLine(points[8], points[9])]
    lines_right.append(gmodel.addLine(points[9], points[10]))
    lines_right.append(gmodel.addLine(points[10], points[11]))
    lines_right.append(gmodel.addLine(points[11], points[8]))

    # lines connecting left to mid
    left_to_mid = [gmodel.addLine(points[0], points[4])]
    left_to_mid.append(gmodel.addLine(points[1], points[5]))
    left_to_mid.append(gmodel.addLine(points[2], points[6]))
    left_to_mid.append(gmodel.addLine(points[3], points[7]))

    # lines connecting mid to right
    mid_to_right = [gmodel.addLine(points[4], points[8])]
    mid_to_right.append(gmodel.addLine(points[5], points[9]))
    mid_to_right.append(gmodel.addLine(points[6], points[10]))
    mid_to_right.append(gmodel.addLine(points[7], points[11]))

    left_loop = gmodel.addCurveLoop(lines_left)
    left_plane = gmodel.addPlaneSurface([left_loop])

    right_loop = gmodel.addCurveLoop(lines_right)
    right_plane = gmodel.addPlaneSurface([right_loop])

    # planes between left and mid
    left_mid_loops = [gmodel.addCurveLoop([lines_left[0], left_to_mid[0], lines_mid[0], left_to_mid[1]])]
    left_mid_loops.append(gmodel.addCurveLoop([lines_left[1], left_to_mid[1], lines_mid[1], left_to_mid[2]]))
    left_mid_loops.append(gmodel.addCurveLoop([lines_left[2], left_to_mid[2], lines_mid[2], left_to_mid[3]]))
    left_mid_loops.append(gmodel.addCurveLoop([lines_left[3], left_to_mid[0], lines_mid[3], left_to_mid[3]]))
    # planes between mid and right
    right_mid_loops = [gmodel.addCurveLoop([lines_mid[0], mid_to_right[1], lines_right[0], mid_to_right[0]])]
    right_mid_loops.append(gmodel.addCurveLoop([lines_mid[1], mid_to_right[1], lines_right[1], mid_to_right[2]]))
    right_mid_loops.append(gmodel.addCurveLoop([lines_mid[2], mid_to_right[2], lines_right[2], mid_to_right[3]]))
    right_mid_loops.append(gmodel.addCurveLoop([lines_mid[3], mid_to_right[0], lines_right[3], mid_to_right[3]]))

    left_mid_planes = [gmodel.addPlaneSurface([loop]) for loop in left_mid_loops]
    right_mid_planes = [gmodel.addPlaneSurface([loop]) for loop in right_mid_loops]

    resolution = args.resolution * 1e-6

    # load centers
    df = scale_x * 470 * pd.read_csv(f'centers/{args.eps_am}.csv')

    centers = []
    for idx in range(df.shape[0]):
        x, y = df.loc[idx, :]
        if (x + Rp) >= LX or (y + Rp) >= LY:
            continue
        if (x - Rp) <= 0 or (y - Rp) <= 0:
            continue
        centers.append((x, y, LZ - Rp))
    n_circles = len(centers)
    pieces = np.linspace(args.lsep, Lz - args.radius, num=11)
    h = scale_z * (pieces[1] - pieces[0])
    h0 = args.se_pos_am_area_frac * h
    h1 = (1 - args.se_pos_am_area_frac) * h
    heights = [-h0, -h1] * 10
    se_pos_am_contact = []
    se_pos_am_no_contact = []

    for piece in pieces[:-1]:
        se_pos_am_no_contact.append(piece * scale_z + 0.5 * h1)
        se_pos_am_contact.append(piece * scale_z + h1 + 0.5 * h0)

    # circle_points = np.zeros((n_circles, 5), dtype=int)
    loop_circles = []
    curved_surfaces = []
    final_disks = []
    for idx, center in enumerate(centers):
        cx, cy, cz = center
        center_point = gmodel.addPoint(*center)
        corners = [
            (cx, cy + Rp, cz),
            (cx - Rp, cy, cz),
            (cx, cy - Rp, cz),
            (cx + Rp, cy, cz),
            ]
        corner_points = [gmodel.addPoint(*p) for p in corners]
        circle_arcs = []

        circle_arcs.append(gmodel.addCircleArc(corner_points[0], center_point, corner_points[1]))
        circle_arcs.append(gmodel.addCircleArc(corner_points[1], center_point, corner_points[2]))
        circle_arcs.append(gmodel.addCircleArc(corner_points[2], center_point, corner_points[3]))
        circle_arcs.append(gmodel.addCircleArc(corner_points[3], center_point, corner_points[0]))

        loop_circle = gmodel.addCurveLoop(circle_arcs)
        loop_circles.append(loop_circle)

        curved_surfs, final_arcs = add_multiple_curve_loops([(1, c) for c in circle_arcs], LZ - Rp, heights, gmodel)
        end_loop = gmodel.addCurveLoop([c[1] for c in final_arcs])
        end_disk = gmodel.addPlaneSurface([end_loop])
        final_disks.append(end_disk)
        curved_surfaces.extend(curved_surfs)

    mid_loop = gmodel.addCurveLoop(lines_mid)
    mid_holey_plane = gmodel.addPlaneSurface([mid_loop] + loop_circles)

    surface_loop_right = gmodel.addSurfaceLoop([c[1] for c in curved_surfaces] + [mid_holey_plane] + final_disks + right_mid_planes + [right_plane], sewing=True)
    vol_right = gmodel.addVolume([surface_loop_right])

    new_surfs = gmodel.getSurfaceLoops(vol_right)[1][0]

    newlyadded = []
    for s in new_surfs:
        com = gmodel.getCenterOfMass(2, s)
        if np.less_equal(com[2], LZ - Rp):
            newlyadded.append(s)

    surface_loop_left = gmodel.addSurfaceLoop([c[1] for c in curved_surfaces] + [mid_holey_plane] + final_disks + left_mid_planes + [left_plane], sewing=True)
    vol_left = gmodel.addVolume([surface_loop_left])
    gmodel.synchronize()
    # gmsh.model.mesh.removeDuplicateElements()
    gmsh.model.addPhysicalGroup(3, [vol_left], markers.electrolyte, "electrolyte")
    gmsh.model.addPhysicalGroup(3, [vol_right], markers.positive_am, "positive_am")
    active_left = []
    right = []
    interface = []
    insulated_se = []
    insulated_am = []

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
            active_left.append(surf[1])
            # if np.any(np.all(np.isclose([com[0], com[1]], left_com_arr, atol=1e-9), axis=1)):
            #     active_left.append(surf[1])
            # else:
            #     inactive_left.append(surf[1])
            # continue
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
            elif np.any(np.isclose(se_pos_am_contact, com[2], atol=1e-9)):
                interface.append(surf[1])
            elif np.any(np.isclose(se_pos_am_no_contact, com[2], atol=1e-9)):
                insulated_am.append(surf[1])
            else:
                pass

    gmsh.model.addPhysicalGroup(2, active_left, markers.left, "left")
    # insulated_se.extend(inactive_left)
    gmsh.model.addPhysicalGroup(2, insulated_se, markers.insulated_electrolyte, "insulated_electrolyte")
    gmsh.model.addPhysicalGroup(2, right, markers.right, "right")
    gmsh.model.addPhysicalGroup(2, insulated_am, markers.insulated_positive_am, "insulated_am")
    gmsh.model.addPhysicalGroup(2, interface, markers.electrolyte_v_positive_am, "electrolyte_positive_am_interface")
    # gmsh.model.occ.synchronize()
    # gmsh.model.occ.healShapes()
    # gmsh.model.mesh.removeDuplicateElements()
    # gmsh.model.mesh.renumberNodes()
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
