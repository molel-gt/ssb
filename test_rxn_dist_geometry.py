#!/usr/bin/env python3
import csv
import json
import os
import pickle
import timeit

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import ufl

from basix.ufl import element
from collections import defaultdict

from dolfinx import cpp, default_scalar_type, fem, io, mesh
from dolfinx.fem import petsc
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import gmshio, VTXWriter
from dolfinx.nls import petsc as petsc_nls
from mpi4py import MPI
from petsc4py import PETSc
import commons, configs, constants, utils


img_id = 11
scale = (800.0 / 470.0) * 1e-6
scale_x = scale
scale_y = scale
scale_z = scale
Lsep = 15 * scale
Lcat = 30 * scale
LX = 470 * scale
LY = 470 * scale
LZ = int((Lsep + Lcat)/scale)
Rp = 6 * scale
eps_se = 0.5
eps_am = 1 - eps_se
markers = commons.Markers()
CELL_TYPES = commons.CellTypes()
resolution = 1

scale_factor = [scale, scale, scale]
dimensions = f'470-470-{LZ}'
print(dimensions)
name_of_study = 'reaction_distribution'
outdir = f"output/{name_of_study}/{dimensions}/{img_id}/{eps_am}/{resolution}"
utils.make_dir_if_missing(outdir)
msh_path = os.path.join(outdir, 'mesh.msh')

comm = MPI.COMM_WORLD
partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
domain, ct, ft = gmshio.read_from_msh(msh_path, comm, partitioner=partitioner)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(tdim, fdim)
# ft_imap = domain.topology.index_map(fdim)
# num_facets = ft_imap.size_local + ft_imap.num_ghosts
# indices = np.arange(0, num_facets)
# values = np.zeros(indices.shape, dtype=np.intc)  # all facets are tagged with zero
# values[ft.indices] = ft.values
# ft = mesh.meshtags(domain, fdim, indices, values)
# ct = mesh.meshtags(domain, domain.topology.dim, ct.indices, ct.values)
left_boundary = ft.find(markers.left)
right_boundary = ft.find(markers.right)