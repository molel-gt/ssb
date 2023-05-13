#!/usr/bin/env python3
# coding: utf-8
import argparse
import gmsh
import itertools
import meshio
import numpy as np

from mpi4py import MPI

import commons, geometry, utils


markers = commons.SurfaceMarkers()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='geometry with slices')
    parser.add_argument('--eps', type=float, help='fraction of lower current collector that is conductive', required=True)
    parser.add_argument('--Lx', help='length', required=True, type=int)
    parser.add_argument('--Ly', help='width', required=True, type=int)
    parser.add_argument("--w", help='slice width along x', nargs='?', const=1, default=0, type=float)
    parser.add_argument("--h", help='slice position along y', nargs='?', const=1, default=0, type=float)
    parser.add_argument("--pos", help='insulator position along x', nargs='?', const=1, default='mid')
    parser.add_argument("--n_pieces", help='insulator position along x', nargs='?', const=1, default=1, type=int)

    args = parser.parse_args()
    pos = args.pos
    n_pieces = args.n_pieces
    eps = args.eps
    Lx = args.Lx
    Ly = args.Ly
    w = args.w / Lx
    h = args.h / Ly
    resolution = 0.00005
    meshname = f'current_constriction/{h:.3}_{w:.3}_pos-{pos}_pieces-{n_pieces}_{eps}'
    utils.make_dir_if_missing('current_constriction')

    gmsh.initialize()
    gmsh.model.add("constriction")
    # gmsh.option.setNumber("General.ExpertMode", 1)
    # gmsh.option.setNumber("Mesh.MeshSizeMin", resolution)
    # gmsh.option.setNumber("Mesh.MeshSizeMax", 0.01)
    dx = Lx * (eps / n_pieces)
    intervals = []
    if np.isclose(n_pieces, 1):
        if np.isclose(eps, 1):
            pass
        else:
            intervals.append((0.5 * Lx - 0.5 * eps * Lx, 0.5 * Lx + 0.5 * eps * Lx))
    else:
        space = Lx * ((0.875 - 0.125) - eps) / (n_pieces - 1)
        for i in range(1, n_pieces + 1):
            intervals.append(
                (0.125 * Lx + (dx + space) * (i - 1),  0.125 * Lx + dx * i + space * (i -1))
                )
    bottom_pts = sorted([v for v in set(itertools.chain(*intervals))])
    points = [
        (0, 0, 0),
        (0, Ly, 0),
        (Lx, Ly, 0),
        (Lx, 0, 0),
        *reversed([(v, 0, 0) for v in bottom_pts])
    ]
    g_points = []
    for p in points:
        g_points.append(
            gmsh.model.occ.addPoint(*p)
        )
    # rotation
    gmsh.model.occ.rotate([(0, p) for p in g_points], 0.5 * Lx, 0.5 * Ly, 0, 0, 0, 1, 0.5 * np.pi)
    channel_lines = []
    left_cc = []
    right_cc = []
    insulated = []
    if n_pieces > 1:
        for i in range(-1, len(g_points)-1):
            line = gmsh.model.occ.addLine(g_points[i], g_points[i + 1])
            channel_lines.append(line)
            if i == 1:
                left_cc.append(line)
            elif i in list(range(4, len(g_points), 2)):
                right_cc.append(line)
            else:
                insulated.append(line)
    else:
        for i in range(-1, len(g_points)-1):
            line = gmsh.model.occ.addLine(g_points[i], g_points[i + 1])
            channel_lines.append(line)
            if i == 1:
                left_cc.append(line)
            elif i == -1:
                right_cc.append(line)
            else:
                insulated.append(line)

    channel_loop = gmsh.model.occ.addCurveLoop(channel_lines)
    channel = gmsh.model.occ.addPlaneSurface((1, channel_loop))

    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(dim=2)
    gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], 1)
    y0_tag = gmsh.model.addPhysicalGroup(1, right_cc, markers.right_cc)
    gmsh.model.setPhysicalName(1, y0_tag, "right_cc")
    yl_tag = gmsh.model.addPhysicalGroup(1, left_cc, markers.left_cc)
    gmsh.model.setPhysicalName(1, yl_tag, "left_cc")
    insulated = gmsh.model.addPhysicalGroup(1, insulated)
    gmsh.model.setPhysicalName(1, insulated, "insulated")
    gmsh.model.occ.synchronize()
    # refinement
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "EdgesList", [yl_tag, y0_tag, insulated])
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", resolution)
    gmsh.model.mesh.field.setNumber(2, "LcMax", 10 * resolution)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 10 * resolution)
    gmsh.model.mesh.field.add("Max", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(f"{meshname}.msh")
    gmsh.finalize()

    mesh_from_file = meshio.read(f"{meshname}.msh")

    triangle_mesh = geometry.create_mesh(mesh_from_file, "triangle")
    meshio.write(f"{meshname}_tria.xdmf", triangle_mesh)
    line_mesh = geometry.create_mesh(mesh_from_file, "line")
    meshio.write(f"{meshname}_line.xdmf", line_mesh)
