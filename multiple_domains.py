#!/usr/bin/env python3
# coding: utf-8
import argparse
import gmsh
import meshio
import numpy as np

from mpi4py import MPI

import geometry, utils


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
    resolution = 0.1
    meshname = f'current_constriction/{h:.3}_{w:.3}_pos-{pos}_pieces-{n_pieces}'
    utils.make_dir_if_missing('current_constriction')

    gmsh.initialize()
    gmsh.model.add("constriction")
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.05)

    channel = gmsh.model.occ.add_rectangle(0, 0, 0, Lx, Ly)
    if h > 0 and w > 0:
        slit_1 = gmsh.model.occ.add_rectangle(0.5 * (1 - w) * Lx, (h - 0.5 * w) * Ly, 0, w * Lx, 0.1 * Ly)
        gmsh.model.occ.cut([(2, channel)], [(2, slit_1)])
    gmsh.model.occ.synchronize()
    surfaces = gmsh.model.getEntities(dim=2)
    gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], 1)
    lines = gmsh.model.getEntities(dim=1)
    top_cc = []
    bottom_cc = []
    insulated = []

    for line in lines:
        com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
        if np.isclose(com[1], 0):
            if np.logical_and(com[0] >= (0.5 * Lx - 0.5 * eps), com[0] <= (0.5 * Lx + 0.5 * eps)):
                bottom_cc.append(line[1])
        elif np.isclose(com[1], Ly):
            top_cc.append(line[1])
        else:
            insulated.append(line[1])
    y0_tag = gmsh.model.addPhysicalGroup(1, bottom_cc)
    gmsh.model.setPhysicalName(1, y0_tag, "bottom_cc")
    yl_tag = gmsh.model.addPhysicalGroup(1, top_cc)
    gmsh.model.setPhysicalName(1, yl_tag, "top_cc")
    insulated = gmsh.model.addPhysicalGroup(1, insulated)
    gmsh.model.setPhysicalName(1, insulated, "insulated")
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.write(f"{meshname}.msh")
    gmsh.finalize()

    mesh_from_file = meshio.read(f"{meshname}.msh")

    triangle_mesh = geometry.create_mesh(mesh_from_file, "triangle")
    meshio.write(f"{meshname}_tria.xdmf", triangle_mesh)
    line_mesh = geometry.create_mesh(mesh_from_file, "line")
    meshio.write(f"{meshname}_line.xdmf", line_mesh)
