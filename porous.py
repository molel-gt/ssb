#!/usr/bin/env python3
# coding: utf-8

import logging
import os
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


img_dir = "/home/emolel3/dev/ssb/mesh/electrolyte/051-051-051_000-000-000"
nodefile = os.path.join(img_dir, "porous.1.node")
facefile = os.path.join(img_dir, "porous.1.face")
stlfile = os.path.join(img_dir, "porous.stl")
Nx = 101
Ny = 101
Nz = 101


def collinear(p0, p1, p2):
    v1 = p0 - p1
    v1_unit = v1 / np.linalg.norm(v1)
    v2 = p2 - p1
    v2_unit = v2 / np.linalg.norm(v2)
    angle = np.arccos(np.dot(v1_unit, v2_unit))

    return np.isclose(angle, 2 * np.pi)


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

cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

for face_idx in range(num_faces):
    local_faces = faces[face_idx]
    p0, p1, p2 = [vertices[p] for p in local_faces]
    for point_id in local_faces:
        vertex = vertices[point_id]
        cube.vectors[face_idx] = point_id
cube.save(stlfile)