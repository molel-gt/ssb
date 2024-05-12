#!/usr/bin/env python3
import argparse
import json
import os

import gmsh
import numpy as np

import utils


class Boundaries:
    def __init__(self):
        pass

    @property
    def left(self):
        return 1

    @property
    def bottom(self):
        return 2

    @property
    def right(self):
        return 3

    @property
    def top(self):
        return 4


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nernst Planck Equation.')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="nernst_planck")
    # parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid',  nargs='?', const=1, default='150-40-0')
    # parser.add_argument('--particle_radius', help='radius of particle in pixel units', nargs='?', const=1, default=10, type=float)
    # parser.add_argument('--well_depth', help='depth of well in pixel units', nargs='?', const=1, default=20, type=float)
    # parser.add_argument('--l_pos', help='thickness of positive electrode in pixel units', nargs='?', const=1, default=75, type=float)
    # parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
    #                     const=1, default='MICRON_TO_METER', type=str)
    parser.add_argument('--resolution', help=f'max resolution resolution', nargs='?', const=1, default=1, type=float)
    parser.add_argument("--refine", help="whether to refine mesh", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    micron = 1e-6
    resolution = args.resolution * micron
    workdir = os.path.join('output', args.name_of_study)
    utils.make_dir_if_missing(workdir)
    output_meshfile = os.path.join(workdir, "mesh.msh")
    output_metafile = os.path.join(workdir, 'geometry.json')
    boundaries = Boundaries()
    points_left = [
        (0, 0, 0),
        (0, 1000 * micron, 0),
    ]
    points_right = [
        (50 * micron, 0, 0),
        (50 * micron, 1000 * micron, 0),
    ]
    points = np.zeros((2, 2), dtype=np.intc)
    lines_horizontal = np.zeros((2, 1), dtype=np.intc)
    lines_vertical = np.zeros((1, 2), dtype=np.intc)
    gmsh.initialize()
    gmsh.model.add('cell')
    points[0, 0] = gmsh.model.occ.addPoint(*points_left[0], meshSize=resolution)
    points[1, 0] = gmsh.model.occ.addPoint(*points_left[1], meshSize=resolution)
    points[0, 1] = gmsh.model.occ.addPoint(*points_right[0], meshSize=resolution)
    points[1, 1] = gmsh.model.occ.addPoint(*points_right[1], meshSize=resolution)
    lines_horizontal[0, 0] = gmsh.model.occ.addLine(points[0, 0], points[0, 1])
    lines_horizontal[1, 0] = gmsh.model.occ.addLine(points[1, 0], points[1, 1])
    lines_vertical[0, 0] = gmsh.model.occ.addLine(points[0, 0], points[1, 0])
    lines_vertical[0, 1] = gmsh.model.occ.addLine(points[0, 1], points[1, 1])
    loop = gmsh.model.occ.addCurveLoop([lines_vertical[0, 0], lines_horizontal[0, 0], lines_vertical[0, 1], lines_horizontal[1, 0]])
    surf = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(1, [lines_vertical[0, 0]], boundaries.left, "left")
    gmsh.model.addPhysicalGroup(1, [lines_horizontal[0, 0]], boundaries.bottom, "bottom")
    gmsh.model.addPhysicalGroup(1, [lines_vertical[0, 1]], boundaries.right, "right")
    gmsh.model.addPhysicalGroup(1, [lines_horizontal[1, 0]], boundaries.top, "top")
    gmsh.model.addPhysicalGroup(2, [surf], 1, "domain")
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
