#!/usr/bin/env python3
import argparse
import json
import os
import warnings

import gmsh
import meshio
import numpy as np

import commons, configs, geometry, utils

warnings.simplefilter('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="reaction_distribution")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid',  nargs='?', const=1, default='75-40-0')
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='MICRON_TO_METER', type=str)
    parser.add_argument('--resolution', help=f'max resolution resolution', nargs='?', const=1, default=1, type=float)
    parser.add_argument("--refine", help="whether to refine the mesh", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    micron = 1e-6
    resolution = np.float16(args.resolution * micron)
    particle_radius = 10
    LX, LY, LZ = [float(val) * micron for val in args.dimensions.split("-")]

    name_of_study = args.name_of_study
    dimensions = args.dimensions
    if not args.refine:
        workdir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], name_of_study, dimensions, "unrefined", f"{args.resolution}")
    else:
        workdir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], name_of_study, dimensions, f"{args.resolution}")
    utils.make_dir_if_missing(workdir)
    output_meshfile = os.path.join(workdir, 'mesh.msh')
    output_metafile = os.path.join(workdir, 'geometry.json')

    markers = commons.Markers()
    points_left = [
    (0, 0, 0),
    (0 * micron, LY, 0)
    ]
    points_right = [
    (LX, 0, 0),
    (LX, LY, 0)
    ]

    interface_2 = [
        (6.5 * particle_radius * micron, 0, 0),

        (6.5 * particle_radius * micron, 1 * particle_radius * micron - micron, 0),
        (6.5 * particle_radius * micron - micron, 1 * particle_radius * micron - micron, 0),
        (6.5 * particle_radius * micron - micron, 1 * particle_radius * micron, 0),

        (2.5 * particle_radius * micron + micron, 1 * particle_radius * micron, 0),
        (2.5 * particle_radius * micron + micron, 1 * particle_radius * micron + micron, 0),
        (2.5 * particle_radius * micron, 1 * particle_radius * micron + micron, 0),

        (2.5 * particle_radius * micron, 3 * particle_radius * micron - micron, 0),
        (2.5 * particle_radius * micron + micron, 3 * particle_radius * micron - micron, 0),
        (2.5 * particle_radius * micron + micron, 3 * particle_radius * micron, 0),

        (6.5 * particle_radius * micron - micron, 3 * particle_radius * micron, 0),
        (6.5 * particle_radius * micron - micron, 3 * particle_radius * micron + micron, 0),
        (6.5 * particle_radius * micron, 3 * particle_radius * micron + micron, 0),

        (6.5 * particle_radius * micron, 4 * particle_radius * micron, 0),
        ]
    interface_points1 = []
    interface_points2 = []
    points_corners = []

    gmsh.initialize()
    gmsh.model.add('lithium-metal')
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)

    if not args.refine:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", resolution)

    points_corners.append(gmsh.model.occ.addPoint(*points_left[0]))
    points_corners.append(gmsh.model.occ.addPoint(*points_left[1]))
    points_corners.append(gmsh.model.occ.addPoint(*points_right[0]))
    points_corners.append(gmsh.model.occ.addPoint(*points_right[1]))
    for p in interface_2:
        interface_points2.append(gmsh.model.occ.addPoint(*p))
    gmsh.model.occ.synchronize()

    lines = []
    lines.append(gmsh.model.occ.addLine(points_corners[0], points_corners[1]))
    lines.append(gmsh.model.occ.addLine(points_corners[2], points_corners[3]))
    lines.append(gmsh.model.occ.addLine(points_corners[0], interface_points2[0]))
    lines.append(gmsh.model.occ.addLine(points_corners[1], interface_points2[-1]))
    lines.append(gmsh.model.occ.addLine(points_corners[2], interface_points2[0]))
    lines.append(gmsh.model.occ.addLine(points_corners[3], interface_points2[-1]))

    interface_lines2 = []
    curr_idx = 0
    for idx in range(len(interface_points2)-1):
        if idx != curr_idx:
            continue
        if curr_idx == 0:
            interface_lines2.append(gmsh.model.occ.addLine(interface_points2[curr_idx], interface_points2[curr_idx+1]))
            curr_idx += 1
        else:
            interface_lines2.append(
                gmsh.model.occ.addCircleArc(
                    interface_points2[curr_idx],
                    interface_points2[curr_idx+1],
                    interface_points2[curr_idx+2]
                    )
                )
            interface_lines2.append(gmsh.model.occ.addLine(interface_points2[curr_idx+2], interface_points2[curr_idx+3]))
            curr_idx += 3

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(1, [lines[0]], markers.left, "left")
    gmsh.model.addPhysicalGroup(1, [lines[1]], markers.right, "right")
    gmsh.model.addPhysicalGroup(1, lines[4:], markers.insulated_positive_am, "insulated_positive_am")
    gmsh.model.addPhysicalGroup(1, lines[2:4], markers.insulated_electrolyte, "insulated_electrolyte")
    gmsh.model.addPhysicalGroup(1, interface_lines2, markers.electrolyte_v_positive_am, "electrolyte_v_positive_am")
    gmsh.model.occ.synchronize()
    electrolyte_loop = gmsh.model.occ.addCurveLoop([lines[idx] for idx in [0, 2, 3]] + interface_lines2)
    pos_am_loop = gmsh.model.occ.addCurveLoop([lines[idx] for idx in [1, 4, 5]] + interface_lines2)
    electrolyte_phase = gmsh.model.occ.addPlaneSurface([electrolyte_loop])
    pos_am_phase = gmsh.model.occ.addPlaneSurface([pos_am_loop])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [electrolyte_phase], markers.electrolyte, "electrolyte")
    gmsh.model.addPhysicalGroup(2, [pos_am_phase], markers.positive_am, "positive_am")
    gmsh.model.occ.synchronize()

    if args.refine:
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", lines + interface_lines2)

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

    gmsh.model.mesh.generate(3)
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
        "resolution": args.resolution,
        "minimum triangle angle (rad)": min(angles),
        "adaptive refine": args.refine,
    }
    with open(output_metafile, "w", encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
