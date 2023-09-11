import os

import numpy as np
import ufl

from dolfinx import cpp, fem, io, mesh, plot
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (div, dx, ds, grad, inner, grad)

import commons, configs, constants, utils


markers = commons.SurfaceMarkers()
phases = commons.Phases()

comm = MPI.COMM_WORLD

workdir = 'mesh/nitsche_diffusivity/61-51-0_000-000-000/'
tria_mesh_filepath = os.path.join(workdir, 'tria.xdmf')
line_mesh_filepath = os.path.join(workdir, 'line.xdmf')
potential_filepath = os.path.join(workdir, 'potential.xdmf')
current_filepath = os.path.join(workdir, 'current.xdmf')

with io.XDMFFile(comm, tria_mesh_filepath, "r") as infile3:
    domain = infile3.read_mesh(cpp.mesh.GhostMode.none, 'Grid')
    ct = infile3.read_meshtags(domain, name="Grid")
domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
with io.XDMFFile(comm, line_mesh_filepath, "r") as infile2:
    ft = infile2.read_meshtags(domain, name="Grid")
meshtags = mesh.meshtags(domain, domain.topology.dim - 1, ft.indices, ft.values)
domaintags = mesh.meshtags(domain, domain.topology.dim, ct.indices, ct.values)

V = fem.FunctionSpace(domain, ("CG", 1))

u_left = fem.Function(V)
u_left.interpolate(lambda x: x[0] - x[0])
u_left.x.scatter_forward()

u_right = fem.Function(V)
u_right.interpolate(lambda x: 0.01 + x[0] - x[0])
u_right.x.scatter_forward()

x = ufl.SpatialCoordinate(domain)
n = ufl.FacetNormal(domain)

# f = -div(grad(1 + x[0] ** 2 + 2 * x[1] ** 2))
f = fem.Constant(domain, PETSc.ScalarType(0.0))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# nitsche terms
h = 2 * ufl.Circumradius(domain)
alpha = 10

# nitsche bilinear
a = inner(grad(u), grad(v)) * dx
a -= inner(n, grad(u)) * v * ds
a += - inner(n, grad(v)) * u * ds
a += alpha / h * inner(u, v) * ds

# nitsche linear
L = inner(f, v) * dx
L += - inner(n, grad(v)) * u_left * ds(markers.left_cc)
L += alpha / h * inner(u_left, v) * ds(markers.left_cc)

L += - inner(n, grad(v)) * u_right * ds(markers.right_cc)
L += alpha / h * inner(u_right, v) * ds(markers.right_cc)

problem = fem.petsc.LinearProblem(a, L)
uh = problem.solve()

with io.XDMFFile(comm, potential_filepath, "w") as outfile:
    outfile.write_mesh(domain)
    outfile.write_function(uh)
