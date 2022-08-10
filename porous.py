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
    nodefile = os.path.join(img_dir, "porous.1.node")
    facefile = os.path.join(img_dir, "porous.1.face")
    stlfile = os.path.join(img_dir, "porous.stl")

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
            for point in points:
                coord = vertices[point]
                triangles.append(np.array(coord))
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