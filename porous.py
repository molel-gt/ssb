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
    facefile = os.path.join(img_dir, "porous.1.face")
    stlfile = os.path.join(img_dir, "porous.stl")
    mshfile = os.path.join(img_dir, "porous_tria.msh")
    

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

    with open(stlfile.replace(".stl", "_left_cc.stl"), "w") as fp1, open(stlfile.replace(".stl", "_right_cc.stl"), "w") as fp2, open(stlfile.replace(".stl", "_insulated.stl"), "w") as fp3:
        fp1.write("solid \n")
        fp2.write("solid \n")
        fp3.write("solid \n")
        for face_idx in range(num_faces):
            points = faces[face_idx]
            triangles = []
            for point in points:
                coord = vertices[point]
                triangles.append(np.array(coord))
            p1, p2, p3 = triangles
            n = np.cross(p3 - p1, p2 - p1)
            n = n / n.sum()
            if np.isclose(p1[1], 0) and np.isclose(p2[1], 0) and np.isclose(p2[1], 0):
                fp1.write("facet normal %f %f %f\n" % tuple(n.tolist()))
                fp1.write("\touter loop\n")
                fp1.write("\t\tvertex %f %f %f\n" % tuple(p1.tolist()))
                fp1.write("\t\tvertex %f %f %f\n" % tuple(p2.tolist()))
                fp1.write("\t\tvertex %f %f %f\n" % tuple(p3.tolist()))
                fp1.write("\tendloop\n")
                fp1.write("endfacet\n")
            elif np.isclose(p1[1], Ny - 1) and np.isclose(p2[1], Ny - 1) and np.isclose(p2[1], Ny - 1):
                fp2.write("facet normal %f %f %f\n" % tuple(n.tolist()))
                fp2.write("\touter loop\n")
                fp2.write("\t\tvertex %f %f %f\n" % tuple(p1.tolist()))
                fp2.write("\t\tvertex %f %f %f\n" % tuple(p2.tolist()))
                fp2.write("\t\tvertex %f %f %f\n" % tuple(p3.tolist()))
                fp2.write("\tendloop\n")
                fp2.write("endfacet\n")
            else:
                fp3.write("facet normal %f %f %f\n" % tuple(n.tolist()))
                fp3.write("\touter loop\n")
                fp3.write("\t\tvertex %f %f %f\n" % tuple(p1.tolist()))
                fp3.write("\t\tvertex %f %f %f\n" % tuple(p2.tolist()))
                fp3.write("\t\tvertex %f %f %f\n" % tuple(p3.tolist()))
                fp3.write("\tendloop\n")
                fp3.write("endfacet\n")
        fp1.write("endsolid \n")
        fp2.write("endsolid \n")
        fp3.write("endsolid \n")
    for marker_suffix in ["_left_cc.stl", "_right_cc.stl", "_insulated.stl"]:
        gmsh.initialize()
        gmsh.merge(stlfile.replace(".stl", marker_suffix))
        gmsh.model.occ.synchronize()
    
        surfaces = gmsh.model.getEntities(dim=2)
        print(surfaces)

        insulated = []
        left_cc = []
        right_cc = []
        for surface in surfaces:
            surf = gmsh.model.addPhysicalGroup(2, [surface[1]])
            gmsh.model.setPhysicalName(2, surf, f"S{surf}")
            print(surf)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write(mshfile.replace(".msh", marker_suffix.replace(".stl", ".msh")))
        gmsh.finalize()
    for marker_suffix in ["_left_cc.stl", "_right_cc.stl", "_insulated.stl"]:
        tria_msh = meshio.read(mshfile.replace(".msh", marker_suffix.replace(".stl", ".msh")))
        print(len(tria_msh))
    tria_mesh = geometry.create_mesh(tria_msh, "triangle")
    meshio.write(f"{img_dir}/tria.xdmf", tria_mesh)