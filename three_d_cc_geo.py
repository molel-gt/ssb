#!/usr/bin/env python3
import argparse
import json
import os

import gmsh
import numpy as np

import commons, configs, utils


def microns(x):
    return x * 1e-6



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Geometry of 3D CC')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="3dcc")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid',  nargs='?', const=1, default='50-250-0')
    parser.add_argument('--radius_minor', help='radius of well', nargs='?', const=1, default=5, type=float)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='MICRON_TO_METER', type=str)
    parser.add_argument('--resolution', help=f'max resolution resolution', nargs='?', const=1, default=1, type=float)
    parser.add_argument('--aspect_ratio', help=f'max resolution resolution', nargs='?', const=1, default=1, type=float)
    parser.add_argument("--refine", help="compute current distribution stats", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    resolution = microns(args.resolution)
    LX, LY, LZ = [microns(float(val)) for val in args.dimensions.split("-")]

    name_of_study = args.name_of_study
    dimensions = args.dimensions
    workdir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], name_of_study, dimensions, str(args.radius_minor), str(args.aspect_ratio), f"{resolution:.1e}")
    utils.make_dir_if_missing(workdir)
    output_meshfile = os.path.join(workdir, 'mesh.msh')
    output_metafile = os.path.join(workdir, 'geometry.json')

    markers = commons.Markers()

    R_minor = microns(args.radius_minor)
    R_major = args.aspect_ratio * R_minor
    L_ncc = microns(25)
    L_sep = microns(25)
    L_cell = L_ncc + L_sep
    W = microns(250)

    # dimensions
    points_left = [
        (0, 0, 0),
        (0, W, 0)
    ]
    points_mid = [
        (L_ncc, 0, 0),
        (L_ncc, W, 0)
    ]
    points_right = [
        (L_cell, 0, 0),
        (L_cell, W, 0),
    ]
    points_arc_top = [
        (L_ncc, 0.5 * W + R_minor + microns(1), 0),
        (L_ncc - microns(1), 0.5 * W + R_minor + microns(1), 0),
        (L_ncc - microns(1), 0.5 * W + R_minor, 0),
    ]
    points_arc_bottom = [
        (L_ncc - microns(1), 0.5 * W - R_minor, 0),
        (L_ncc - microns(1), 0.5 * W - R_minor - microns(1), 0),
        (L_ncc, 0.5 * W - R_minor - microns(1), 0),
    ]

    points_ellipse_center = (L_ncc - microns(1), 0.5 * W, 0)
    points_ellipse_arc = [
        (L_ncc - microns(1), 0.5 * W + R_minor, 0),
        (L_ncc - microns(1) - R_major, 0.5 * W, 0),
        (L_ncc - microns(1), 0.5 * W - R_minor, 0),
    ]

    points_arr = np.zeros((2, 3), dtype=np.int32)
    gmsh.initialize()
    gmsh.model.add('3dcc_well')
    points_arr[0, 0] = gmsh.model.occ.addPoint(*points_left[0])
    points_arr[1, 0] = gmsh.model.occ.addPoint(*points_left[1])

    points_arr[0, 1] = gmsh.model.occ.addPoint(*points_mid[0])
    points_arr[1, 1] = gmsh.model.occ.addPoint(*points_mid[1])

    points_arr[0, 2] = gmsh.model.occ.addPoint(*points_right[0])
    points_arr[1, 2] = gmsh.model.occ.addPoint(*points_right[1])

    points_arc_t = []
    for p in points_arc_top:
        points_arc_t.append(gmsh.model.occ.addPoint(*p))

    points_arc_b = []
    for p in points_arc_bottom:
        points_arc_b.append(gmsh.model.occ.addPoint(*p))

    ellipse_center = gmsh.model.occ.addPoint(*points_ellipse_center)
    ellipse_center_2 = gmsh.model.occ.addPoint(L_ncc - microns(1) - 0.5 * R_major, 0.5 * W, 0)
    ellipse_end = gmsh.model.occ.addPoint(L_ncc - microns(1) - R_major, 0.5 * W, 0)
    points_ellipse = []
    for p in points_ellipse_arc:
        points_ellipse.append(gmsh.model.occ.addPoint(*p))

    external_lines = []
    external_lines.append(gmsh.model.occ.addLine(*points_arr[0, [0, 1]]))
    external_lines.append(gmsh.model.occ.addLine(*points_arr[0, [1, 2]]))
    external_lines.append(gmsh.model.occ.addLine(*points_arr[[0, 1], 2]))
    external_lines.append(gmsh.model.occ.addLine(*points_arr[1, [1, 2]]))
    external_lines.append(gmsh.model.occ.addLine(*points_arr[1, [0, 1]]))
    external_lines.append(gmsh.model.occ.addLine(*points_arr[[0, 1], 0]))
    # gmsh.model.occ.synchronize()

    lines_mid = []
    three_d_cc_well = []
    lines_mid.append(gmsh.model.occ.addLine(points_arr[1, 1], points_arc_t[0]))
    lines_mid.append(gmsh.model.occ.addCircleArc(*points_arc_t))
    three_d_cc_well.append(gmsh.model.occ.addEllipseArc(points_arc_t[-1], ellipse_center, ellipse_center_2, ellipse_end))
    three_d_cc_well.append(gmsh.model.occ.addEllipseArc(points_arc_b[0], ellipse_center, ellipse_center_2, ellipse_end))
    three_d_cc_well.append(gmsh.model.occ.addCircleArc(*points_arc_b))
    lines_mid.extend(three_d_cc_well)
    lines_mid.append(gmsh.model.occ.addLine(points_arr[0, 1], points_arc_b[-1]))
    gmsh.model.occ.synchronize()
    loop_ncc = gmsh.model.occ.addCurveLoop([external_lines[0], external_lines[5], external_lines[4]] + lines_mid)
    ncc_surf = gmsh.model.occ.addPlaneSurface([loop_ncc])
    loop_electrolyte = gmsh.model.occ.addCurveLoop([external_lines[1], external_lines[2], external_lines[3]] + lines_mid)
    electrolyte_surf = gmsh.model.occ.addPlaneSurface([loop_electrolyte])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [ncc_surf], markers.negative_cc, "negative cc")
    gmsh.model.addPhysicalGroup(2, [electrolyte_surf], markers.electrolyte, "electrolyte")
    gmsh.model.addPhysicalGroup(1, [external_lines[5]], markers.left, "left")
    gmsh.model.addPhysicalGroup(1, [external_lines[2]], markers.right, "right")

    gmsh.model.addPhysicalGroup(1, [external_lines[0], external_lines[4]], markers.insulated_negative_cc, "insulated negative cc")
    gmsh.model.addPhysicalGroup(1, [external_lines[1], external_lines[3]], markers.insulated_electrolyte, "insulated electrolyte")
    gmsh.model.addPhysicalGroup(1, lines_mid, markers.negative_am_v_electrolyte, "negative am - electrolyte interface")
    gmsh.model.addPhysicalGroup(1, three_d_cc_well, markers.three_d_cc_well, "3D CC Well")
    gmsh.model.occ.synchronize()

    if args.refine:
            gmsh.model.mesh.field.add("Distance", 1)
            gmsh.model.mesh.field.setNumbers(1, "CurvesList", [external_lines[5], external_lines[2]] + lines_mid)

            gmsh.model.mesh.field.add("Threshold", 2)
            gmsh.model.mesh.field.setNumber(2, "IField", 1)
            gmsh.model.mesh.field.setNumber(2, "LcMin", resolution/5)
            gmsh.model.mesh.field.setNumber(2, "LcMax", resolution)
            gmsh.model.mesh.field.setNumber(2, "DistMin", resolution)
            gmsh.model.mesh.field.setNumber(2, "DistMax", 5 * resolution)

            gmsh.model.mesh.field.add("Max", 5)
            gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
            gmsh.model.mesh.field.setAsBackgroundMesh(5)
            gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(output_meshfile)
    

    surfs = gmsh.model.getEntities(2)
    angles = []
    for surf in surfs:
        _, _triangles, _nodes = gmsh.model.mesh.getElements(surf[0], surf[1])
        triangles = _triangles[0]
        nodes = _nodes[0].reshape(triangles.shape[0], 3)
        for idx in range(triangles.shape[0]):
            p1, p2, p3 = [gmsh.model.mesh.getNode(nodes[idx, i])[0] for i in range(3)]
            _angles = utils.compute_angles_in_triangle(p1, p2, p3)
            angles += _angles
    print(f"Minimum angle in triangles is {np.rad2deg(min(angles)):.2f} degrees")
    gmsh.finalize()

    metadata = {
        "resolution": resolution,
        "minimum triangle angle (rad)": min(angles),
        "adaptive refine": args.refine,
    }
    with open(output_metafile, "w", encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
