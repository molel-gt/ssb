#!/usr/bin/env python3
# coding: utf-8

import logging
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import argparse
import gmsh
import meshio
import numpy as np

from itertools import groupby
from operator import itemgetter
from stl import mesh

import geometry, utils


FORMAT = '%(asctime)s: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__file__)
logger.setLevel('INFO')


if __name__ == '__main__':
    img_dir = sys.argv[1]
    grid_info = sys.argv[2]
    Nx, Ny, Nz = [int(v) for v in grid_info.split("-")]
    nodefile = os.path.join(img_dir, "porous.1.node")
    tetsfile = os.path.join(img_dir, "porous.1.ele")
    facefile = os.path.join(img_dir, "porous.1.face")
    stlfile = os.path.join(img_dir, "porous.stl")
    mshfile = os.path.join(img_dir, "porous_tria.msh")
    tets_points = set()
    with open(tetsfile, "r") as fp:
        line = 0
        for row in fp.readlines():
            line += 1
            if row.startswith("#"):
                continue
            if line == 1:
                num_vertices = int(row.split()[0])
                vertices = np.zeros((num_vertices, 3), dtype=int)
                continue
            tet_id, a, b, c, d = [float(v) for v in row.split()]
            tets_points.update(set([a, b, c, d]))
    with open(nodefile, 'r') as fp:
        line = 0
        for row in fp.readlines():
            line += 1
            if row.startswith("#"):
                continue
            if line == 1:
                num_vertices = int(row.split()[0])
                vertices = np.zeros((num_vertices, 3), dtype=int)
                continue
            point_id, x, y, z = [float(v) for v in row.split()]
            vertices[int(point_id)] = [x, y, z]
    with open(facefile, "r") as fp:
        line = 0
        for row in fp.readlines():
            line += 1
            if row.startswith("#"):
                continue
            if line == 1:
                num_faces = int(row.split()[0])
                faces = np.zeros((num_faces, 3), dtype=int)
                continue
            face_id, p1, p2, p3 = [int(v) for v in row.split()]
            faces[face_id] = [p1, p2, p3]

    with open(stlfile, "w") as fp:
        fp.write("solid \n")
        for face_idx in range(num_faces):
            points = faces[face_idx]
            triangles = []
            all_in_tet = True
            for point in points:
                if point not in tets_points:
                    all_in_tet = False
                coord = vertices[point]
                triangles.append(np.array(coord))
            if not all_in_tet:
                continue
            p1, p2, p3 = triangles
            n = np.cross(p3 - p2, p2 - p1)
            n = n / n.sum()
            fp.write("facet normal %f %f %f\n" % tuple(n.tolist()))
            fp.write("\touter loop\n")
            fp.write("\t\tvertex %f %f %f\n" % tuple(p1.tolist()))
            fp.write("\t\tvertex %f %f %f\n" % tuple(p2.tolist()))
            fp.write("\t\tvertex %f %f %f\n" % tuple(p3.tolist()))
            fp.write("\tendloop\n")
            fp.write("endfacet\n")
        fp.write("endsolid \n")

    gmsh.initialize()
    gmsh.merge(stlfile)
    model = gmsh.model
    factory = model.occ
    factory.synchronize()

    surfaces = model.getEntities(dim=2)

    insulated = []
    left_cc = []
    right_cc = []
    for surface in surfaces:
        surf = model.addPhysicalGroup(2, [surface[1]])
        model.setPhysicalName(2, surf, f"S{surf}")
    model.geo.synchronize()
    model.mesh.generate(2)
    gmsh.write(mshfile)
    gmsh.finalize()
    tria_msh = meshio.read(mshfile)
    tria_mesh = geometry.create_mesh(tria_msh, "triangle")
    meshio.write(f"{img_dir}/tria.xdmf", tria_mesh)