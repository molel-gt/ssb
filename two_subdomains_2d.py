import os

import dolfinx
import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
import ufl
import warnings

from dolfinx import cpp, default_scalar_type, fem, io, mesh, nls, plot
from dolfinx.fem import petsc
from dolfinx.io import VTXWriter
from dolfinx.nls import petsc as petsc_nls
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 dot, div, dx, ds, dS, grad, inner, grad, avg, jump)

import commons, geometry, utils

warnings.simplefilter('ignore')

encoding = io.XDMFFile.Encoding.HDF5
adaptive_refine = True
micron = 1e-6
markers = commons.Markers()
LX = 200 * micron
LY = 250 * micron
points = [
    (0, 0, 0),
    (0.5 * LX, 0, 0),
    (LX, 0, 0),
    (LX, LY, 0),
    (0.5 * LX, LY, 0),
    (0, LY, 0),
]
gpoints = []
lines = []
# workdir = "output/full_cell/165-825-0/1/" 
workdir = "output/subdomains_dg"
utils.make_dir_if_missing(workdir)
output_meshfile = os.path.join(workdir, 'mesh.msh')
tria_meshfile = os.path.join(workdir, "tria.xdmf")
line_meshfile = os.path.join(workdir, "line.xdmf")

gmsh.initialize()
gmsh.model.add('full-cell')
# gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1 * micron)
# gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5 * micron)
for idx, p in enumerate(points):
    gpoints.append(
        gmsh.model.occ.addPoint(*p)
    )
gmsh.model.occ.synchronize()
gmsh.model.occ.synchronize()
for idx in range(0, len(points)-1):
    lines.append(
        gmsh.model.occ.addLine(gpoints[idx], gpoints[idx+1])
    )
lines.append(
    gmsh.model.occ.addLine(gpoints[-1], gpoints[0])
)
lines.append(
    gmsh.model.occ.addLine(gpoints[1], gpoints[4])
)

gmsh.model.occ.synchronize()
ltag = gmsh.model.addPhysicalGroup(1, [lines[-2]], markers.left)
gmsh.model.setPhysicalName(1, ltag, "left")
rtag = gmsh.model.addPhysicalGroup(1, [lines[2]], markers.right)
gmsh.model.setPhysicalName(1, rtag, "right")
evptag = gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [-1]], markers.electrolyte_v_positive_am)
gmsh.model.setPhysicalName(1, evptag, "electrolyte_v_positive_am")
ietag = gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [0, 4]], markers.insulated_electrolyte)
gmsh.model.setPhysicalName(1, ietag, "insulated_electrolyte")
ipamtag = gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [1, 3]], markers.insulated_positive_am)
gmsh.model.setPhysicalName(1, ipamtag, "insulated_positive_am")
gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [0, 1, 3, 4]], markers.insulated)
gmsh.model.occ.synchronize()
se_loop = gmsh.model.occ.addCurveLoop([lines[idx] for idx in [0, 6, 4, 5]])
pe_loop = gmsh.model.occ.addCurveLoop([lines[idx] for idx in [1, 2, 3, 6]])
gmsh.model.occ.synchronize()
se_phase = gmsh.model.occ.addPlaneSurface([se_loop])
pe_phase = gmsh.model.occ.addPlaneSurface([pe_loop])
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(2, [se_phase], markers.electrolyte)
gmsh.model.addPhysicalGroup(2, [pe_phase], markers.positive_am)
gmsh.model.occ.synchronize()

if adaptive_refine:
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "EdgesList", [lines[idx] for idx in [-1, -2, 0, 1, 2, 3, 4, 5, 6]])
    
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", 0.1 * micron)
    gmsh.model.mesh.field.setNumber(2, "LcMax", 1 * micron)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1 * micron)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 1 * micron)
    
    gmsh.model.mesh.field.add("Max", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)
gmsh.model.occ.synchronize()


gmsh.model.mesh.generate(2)
gmsh.write(output_meshfile)
gmsh.finalize()

mesh_2d = meshio.read(output_meshfile)
tria_mesh = geometry.create_mesh(mesh_2d, "triangle")
meshio.write(tria_meshfile, tria_mesh)
line_mesh = geometry.create_mesh(mesh_2d, "line")
meshio.write(line_meshfile, line_mesh)