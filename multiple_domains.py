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
    parser.add_argument('--Lx', help='length', required=True)
    parser.add_argument('--Ly', help='width', required=True)
    parser.add_argument("--w", help='slice width along x', nargs='?', const=1, default=10, type=float)
    parser.add_argument("--h", help='slice position along y', nargs='?', const=1, default=0.5, type=float)

    args = parser.parse_args()
    coverage = args.coverage
    meshname = args.meshname
    Lx = args.Lx
    Ly = args.Ly
    w = args.w / Lx
    h = args.h / Ly
    resolution = 0.1
    meshname = 'current_constriction/{%0.3fh}-{%0.3fw}'
    utils.make_dir_if_missing('current_constriction')

    gmsh.initialize()
    gmsh.model.add("constriction")
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.01)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.01)

    channel = gmsh.model.occ.add_rectangle(0, 0, 0, Lx, Ly)
    slit_1 = gmsh.model.occ.add_rectangle(0.5 * (1 - w) * Lx, (h - 0.5 * w) * Ly, 0, w * Lx, w * Ly)
    gmsh.model.occ.cut([(2, channel)], [(2, slit_1)])
    gmsh.model.occ.synchronize()
    surfaces = gmsh.model.getEntities(dim=2)
    gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], 1)

    gmsh.model.mesh.generate(2)
    gmsh.write(f"{meshname}.msh")
    gmsh.finalize()

    mesh_from_file = meshio.read(f"{meshname}.msh")

    triangle_mesh = geometry.create_mesh(mesh_from_file, "triangle")
    meshio.write(f"{meshname}.xdmf", triangle_mesh)
