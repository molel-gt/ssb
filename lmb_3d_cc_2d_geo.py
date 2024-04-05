#!/usr/bin/env python3
import argparse
import os
import warnings

import gmsh
import numpy as np

import commons, configs, geometry, utils

warnings.simplefilter('ignore')


def build_rectangular_curve(coord_start=(10e-6, 0, 0), step_width=10e-6, step_length=40e-6, length=500e-6):
    points = [coord_start, (coord_start[0], coord_start[1] + 0.5 * step_length, coord_start[2])]
    n_pieces = int(length / step_length)
    if n_pieces < 2:
        raise ValueError("Undefined for less than 2 pieces")

    for idx in range(n_pieces):
        if idx % 2 == 1:
            continue
        x, y, z = points[-1]
        to_add = [
        (x + step_width, y, z),
        (x + step_width, y + step_length, z),
        (x, y + step_length, z),
        (x, y + 2 * step_length, z),
        ]
        for p in to_add:
            if p[1] < coord_start[2] + length:
                points.append(p)
            else:
                points.append((p[0], length, p[2]))

    return points


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
    points_left = [
    (0, 0, 0),
    (0, LY, 0)
    ]
    points_right = [
    (LX, 0, 0),
    (LX, LY, 0)
    ]
    coord_start1 = (20e-6, 0, 0)
    coord_start2 = (75e-6, 0, 0)

    # interface_1 = build_rectangular_curve(coord_start=coord_start1, step_length=step_length, step_width=step_width1, length=LY)
    # interface_2 = build_rectangular_curve(coord_start=coord_start2, step_length=step_length, step_width=step_width2, length=LY)
    interface_1 = [
        (3 * args.particle_radius * micron, 0, 0),
        # arc points
        (3 * args.particle_radius * micron, 1 * args.particle_radius * micron - micron, 0),
        (3 * args.particle_radius * micron - micron, 1 * args.particle_radius * micron - micron, 0),
        (3 * args.particle_radius * micron - micron, 1 * args.particle_radius * micron, 0),

        # arc points
        (1 * args.particle_radius * micron + micron, 1 * args.particle_radius * micron, 0),
        (1 * args.particle_radius * micron + micron, 1 * args.particle_radius * micron + micron, 0),
         (1 * args.particle_radius * micron, 1 * args.particle_radius * micron + micron, 0),

        # arc points
        (1 * args.particle_radius * micron, 3 * args.particle_radius * micron - micron, 0),
        (1 * args.particle_radius * micron + micron, 3 * args.particle_radius * micron - micron, 0),
        (1 * args.particle_radius * micron + micron, 3 * args.particle_radius * micron, 0),

        # arc points
        (3 * args.particle_radius * micron - micron, 3 * args.particle_radius * micron, 0),
        (3 * args.particle_radius * micron - micron, 3 * args.particle_radius * micron + micron, 0),
        (3 * args.particle_radius * micron, 3 * args.particle_radius * micron + micron, 0),

        (3 * args.particle_radius * micron, 4 * args.particle_radius * micron, 0),
        ]
    interface_2 = [
        (14 * args.particle_radius * micron, 0, 0),

        (14 * args.particle_radius * micron, 1 * args.particle_radius * micron - micron, 0),
        (14 * args.particle_radius * micron - micron, 1 * args.particle_radius * micron - micron, 0),
        (14 * args.particle_radius * micron - micron, 1 * args.particle_radius * micron, 0),

        (7.5 * args.particle_radius * micron + micron, 1 * args.particle_radius * micron, 0),
        (7.5 * args.particle_radius * micron + micron, 1 * args.particle_radius * micron + micron, 0),
        (7.5 * args.particle_radius * micron, 1 * args.particle_radius * micron + micron, 0),

        (7.5 * args.particle_radius * micron, 3 * args.particle_radius * micron - micron, 0),
        (7.5 * args.particle_radius * micron + micron, 3 * args.particle_radius * micron - micron, 0),
        (7.5 * args.particle_radius * micron + micron, 3 * args.particle_radius * micron, 0),

        (14 * args.particle_radius * micron - micron, 3 * args.particle_radius * micron, 0),
        (14 * args.particle_radius * micron - micron, 3 * args.particle_radius * micron + micron, 0),
        (14 * args.particle_radius * micron, 3 * args.particle_radius * micron + micron, 0),

        (14 * args.particle_radius * micron, 4 * args.particle_radius * micron, 0),
        ]
    interface_points1 = []
    interface_points2 = []
    points_corners = []

    gmsh.initialize()
    gmsh.model.add('lithium-metal')
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", resolution)
    # gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 1)
    # gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 1)
    # gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    # gmsh.option.setNumber('Mesh.ColorCarousel', 2)
    gmsh.option.setNumber('Mesh.Optimize', 1)
    gmsh.option.setNumber('Mesh.Algorithm', 5)
    gmsh.option.setNumber('Mesh.OptimizeThreshold', 0.5)

    # points_corners.append(gmsh.model.occ.addPoint(*points_left[0]))
    # points_corners.append(gmsh.model.occ.addPoint(*points_left[1]))
    points_corners += [np.nan, np.nan]
    points_corners.append(gmsh.model.occ.addPoint(*points_right[0]))
    points_corners.append(gmsh.model.occ.addPoint(*points_right[1]))
    for p in interface_1:
        interface_points1.append(gmsh.model.occ.addPoint(*p))
    for p in interface_2:
        interface_points2.append(gmsh.model.occ.addPoint(*p))
    gmsh.model.occ.synchronize()

    lines = []
    lines += [np.nan]
    # lines.append(gmsh.model.occ.addLine(points_corners[0], points_corners[1]))
    lines.append(gmsh.model.occ.addLine(points_corners[2], points_corners[3]))
    lines += [np.nan]
    # lines.append(gmsh.model.occ.addLine(points_corners[0], interface_points1[0]))
    lines += [np.nan]
    # lines.append(gmsh.model.occ.addLine(points_corners[1], interface_points1[-1]))
    lines.append(gmsh.model.occ.addLine(points_corners[2], interface_points2[0]))
    lines.append(gmsh.model.occ.addLine(points_corners[3], interface_points2[-1]))

    lines.append(gmsh.model.occ.addLine(interface_points1[0], interface_points2[0]))
    lines.append(gmsh.model.occ.addLine(interface_points1[-1], interface_points2[-1]))
    
    interface_lines1 = []
    interface_lines2 = []

    curr_idx = 0
    for idx in range(len(interface_points1)-1):
        if idx != curr_idx:
            continue
        if curr_idx == 0:
            interface_lines1.append(gmsh.model.occ.addLine(interface_points1[curr_idx], interface_points1[curr_idx+1]))
            curr_idx += 1
            continue
        else:
            interface_lines1.append(
                gmsh.model.occ.addCircleArc(
                    interface_points1[curr_idx],
                    interface_points1[curr_idx+1],
                    interface_points1[curr_idx+2]
                    )
                )
            interface_lines1.append(gmsh.model.occ.addLine(interface_points1[curr_idx+2], interface_points1[curr_idx+3]))
            curr_idx += 3
            continue
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

    # gmsh.model.addPhysicalGroup(1, [lines[0]], markers.left, "left")
    gmsh.model.addPhysicalGroup(1, [lines[1]], markers.right, "right")
    # gmsh.model.addPhysicalGroup(1, lines[2:4], markers.insulated_negative_cc, "insulated_negative_cc")
    gmsh.model.addPhysicalGroup(1, lines[4:6], markers.insulated_positive_am, "insulated_positive_am")
    gmsh.model.addPhysicalGroup(1, lines[6:8], markers.insulated_electrolyte, "insulated_electrolyte")
    gmsh.model.addPhysicalGroup(1, interface_lines1, markers.left, "negative_cc_v_negative_am")
    gmsh.model.addPhysicalGroup(1, interface_lines2, markers.electrolyte_v_positive_am, "electrolyte_v_positive_am")
    gmsh.model.occ.synchronize()

    # neg_cc_loop = gmsh.model.occ.addCurveLoop([lines[idx] for idx in [0, 2]] + interface_lines1 + [lines[3]])
    electrolyte_loop = gmsh.model.occ.addCurveLoop([lines[6]] + interface_lines1 + [lines[7]] + interface_lines2)
    pos_am_loop = gmsh.model.occ.addCurveLoop([lines[idx] for idx in [1, 4]] + interface_lines2 + [lines[5]])
    # neg_cc_phase = gmsh.model.occ.addPlaneSurface([neg_cc_loop])
    electrolyte_phase = gmsh.model.occ.addPlaneSurface([electrolyte_loop])
    pos_am_phase = gmsh.model.occ.addPlaneSurface([pos_am_loop])
    gmsh.model.occ.synchronize()
    # gmsh.model.addPhysicalGroup(2, [neg_cc_phase], markers.negative_cc, "negative_cc")
    gmsh.model.addPhysicalGroup(2, [electrolyte_phase], markers.electrolyte, "electrolyte")
    gmsh.model.addPhysicalGroup(2, [pos_am_phase], markers.positive_am, "positive_am")
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "EdgesList", [lines[1]] + interface_lines1 + interface_lines2)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", resolution/20)
    gmsh.model.mesh.field.setNumber(2, "LcMax", resolution)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.01 * micron)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 2 * micron)

    gmsh.model.mesh.field.add("Max", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)
    # elementType = gmsh.model.mesh.getElementType("triangle", 2)
    # elementTypes = gmsh.model.mesh.getElementTypes(dim=2)
    # print(elementType, elementTypes)
    # print(gmsh.model.mesh.getElementQualities(elementTypes, qualityName='angleShape'))
    
    _, eleTags , _ = gmsh.model.mesh.getElements(dim=2)
    # q = gmsh.model.mesh.getElementQualities(eleTags[0], "minSICN")
    q = gmsh.model.mesh.getElementQualities(eleTags[0], "angleShape")
    for vv in zip(eleTags[0], q):
        print(vv[1])
    # gmsh.plugin.setNumber("AnalyseMeshQuality", "ICNMeasure", 1.)
    # gmsh.plugin.setNumber("AnalyseMeshQuality", "CreateView", 1.)
    # t = gmsh.plugin.run("AnalyseMeshQuality")
    # dataType, tags, data, time, numComp = gmsh.view.getModelData(t, 0)
    # print('ICN for element {0} = {1}'.format(tags[0], data[0]))

    gmsh.write(output_meshfile)
    gmsh.finalize()