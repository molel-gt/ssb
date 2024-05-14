#!/usr/bin/env python3
import argparse
import json
import os
import warnings

import gmsh
import numpy as np

import commons, configs, geometry, utils

warnings.simplefilter('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="lithium_metal_3d_cc_2d")
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

    name_of_study = args.name_of_study
    step_length = 2 * args.particle_radius * micron
    step_width1 = args.well_depth * micron
    step_width2 = (args.l_pos - 2 * args.particle_radius) * micron
    dimensions = args.dimensions
    dimensions_ii = f'{int(step_width1/micron)}-{int(step_width2/micron)}-{int(step_length/micron)}'
    workdir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], name_of_study, dimensions, dimensions_ii, f"{resolution:.1e}")
    utils.make_dir_if_missing(workdir)
    output_meshfile = os.path.join(workdir, 'mesh.msh')
    output_metafile = os.path.join(workdir, 'geometry.json')

    markers = commons.Markers()

    points_left = [
    (0, 0, 0),
    (0, 0.25 * LY, 0),
    (0, 0.75 * LY, 0),
    (0, LY, 0)
    ]
    points_1 = [
    (micron * args.particle_radius, 0, 0),
    (micron * args.particle_radius, 0.25 * LY, 0),
    (micron * args.particle_radius, 0.75 * LY, 0),
    (micron * args.particle_radius, LY, 0)
    ]
    points_2 = [
    (2 * micron * args.particle_radius, 0, 0),
    (2 * micron * args.particle_radius, 0.25 * LY, 0),
    (2 * micron * args.particle_radius, 0.75 * LY, 0),
    (2 * micron * args.particle_radius, LY, 0)
    ]
    points_3 = [
    (0.5 * LX, 0, 0),
    (0.5 * LX, 0.25 * LY, 0),
    (0.5 * LX, 0.75 * LY, 0),
    (0.5 * LX, LY, 0)
    ]
    points_4 = [
    (LX - micron * args.particle_radius, 0, 0),
    (LX - micron * args.particle_radius, 0.25 * LY, 0),
    (LX - micron * args.particle_radius, 0.75 * LY, 0),
    (LX - micron * args.particle_radius, LY, 0)
    ]
    points_right = [
    (LX, 0, 0),
    (LX, 0.25 * LY, 0),
    (LX, 0.75 * LY, 0),
    (LX, LY, 0)
    ]

    points = np.zeros((4, 6), dtype=np.intc)
    lines_vertical = np.zeros((3, 6), dtype=np.intc)
    lines_horizontal = np.zeros((4, 5), dtype=np.intc)
    loops = np.zeros((3, 5), dtype=np.intc)
    surfaces = np.zeros((3, 5), dtype=np.intc)

    gmsh.initialize()
    gmsh.model.add('lithium-metal')

    for idx, p in enumerate(points_left):
        points[idx, 0] = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    for idx, p in enumerate(points_1):
        points[idx, 1] = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    for idx, p in enumerate(points_2):
        points[idx, 2] = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    for idx, p in enumerate(points_3):
        points[idx, 3] = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    for idx, p in enumerate(points_4):
        points[idx, 4] = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    for idx, p in enumerate(points_right):
        points[idx, 5] = gmsh.model.occ.addPoint(*p, meshSize=resolution)

    for col in range(6):
        for row in range(3):
            lines_vertical[row, col] = gmsh.model.occ.addLine(points[row, col], points[row + 1, col])

    for row in range(4):
        for col in range(5):
            lines_horizontal[row, col] = gmsh.model.occ.addLine(points[row, col], points[row, col + 1])

    gmsh.model.occ.synchronize()
    for row in range(3):
        for col in range(5):
            loops[row, col] = gmsh.model.occ.addCurveLoop(
                [lines_horizontal[row, col],
                lines_vertical[row, col+1],
                lines_horizontal[row+1, col],
                lines_vertical[row, col]]
                )
            surfaces[row, col] = gmsh.model.occ.addPlaneSurface([loops[row, col]])
            gmsh.model.occ.synchronize()

    # gmsh.model.addPhysicalGroup(1, [lines[0]], markers.left, "left")
    gmsh.model.addPhysicalGroup(1, list(lines_vertical[:, -1]), markers.right, "right")
    # gmsh.model.addPhysicalGroup(1, lines[2:4], markers.insulated_negative_cc, "insulated_negative_cc")
    gmsh.model.addPhysicalGroup(1, [lines_horizontal[0, idx] for idx in [4]] + [lines_horizontal[-1, idx] for idx in [4]], markers.insulated_positive_am, "insulated_positive_am")
    gmsh.model.addPhysicalGroup(1, [lines_horizontal[0, idx] for idx in [2, 3]] + [lines_horizontal[-1, idx] for idx in [2, 3]], markers.insulated_electrolyte, "insulated_electrolyte")
    gmsh.model.addPhysicalGroup(1, [lines_vertical[1, 1]] + [lines_vertical[idx, 2] for idx in [0, 2]] + list(lines_horizontal[[1, 2], 1]), markers.left, "negative_cc_v_negative_am")
    gmsh.model.addPhysicalGroup(1, [lines_vertical[1, 3]] + [lines_vertical[idx, 4] for idx in [0, 2]] + list(lines_horizontal[1, -2:-1]) + list(lines_horizontal[2, -2:-1]), markers.electrolyte_v_positive_am, "electrolyte_v_positive_am")
    gmsh.model.occ.synchronize()

    gmsh.model.occ.synchronize()
    
    electrolyte_surfs = [surfaces[1, 1]] + list(surfaces[:, 2]) + [surfaces[0, 3]] + [surfaces[-1, 3]]
    positive_am_surfs = list(surfaces[:, -1]) + [surfaces[1, -2]]
    valid_surfs = electrolyte_surfs + positive_am_surfs
    # gmsh.model.addPhysicalGroup(2, [neg_cc_phase], markers.negative_cc, "negative_cc")
    gmsh.model.addPhysicalGroup(2, electrolyte_surfs, markers.electrolyte, "electrolyte")
    gmsh.model.addPhysicalGroup(2, positive_am_surfs, markers.positive_am, "positive_am")
    gmsh.model.occ.synchronize()

    refine_boundaries = [lines_vertical[1, 3]] + [lines_vertical[idx, 4] for idx in [0, 2]] + list(lines_horizontal[1, -2:-1]) + list(lines_horizontal[2, -2:-1])
    refine_boundaries += list(lines_vertical[:, -1])
    refine_boundaries += [lines_vertical[1, 1]] + [lines_vertical[idx, 2] for idx in [0, 2]] + list(lines_horizontal[[1, 2], 1])

    gmsh.model.mesh.setTransfiniteAutomatic([(2, s) for s in valid_surfs], cornerAngle=np.pi/4, recombine=False)
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
