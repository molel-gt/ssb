#!/usr/bin/env python3

import os

import argparse
import dolfinx
import numpy as np
import ufl
from dolfinx.fem import (dirichletbc as DirichletBC, Function, FunctionSpace)
from dolfinx.fem import Constant, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from ufl import ds, dx, grad, inner

import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run simulation..')
    parser.add_argument('--grid_info', help='Nx-Ny-Nz', required=True)
    parser.add_argument('--origin', default=(0, 0, 0), help='where to extract grid from')

    args = parser.parse_args()
    if isinstance(args.origin, str):
        origin = tuple(map(lambda v: int(v), args.origin.split(",")))
    else:
        origin = args.origin
    origin_str = "_".join([str(v) for v in origin])
    grid_info = args.grid_info
    Lx = int(grid_info.split("-")[0]) - 1
    working_dir = os.path.abspath(os.path.dirname(__file__))
    meshes_dir = os.path.join(working_dir, 'mesh')
    output_dir = os.path.join(working_dir, 'output')
    utils.make_dir_if_missing(meshes_dir)
    utils.make_dir_if_missing(output_dir)
    tetr_mesh_path = os.path.join(meshes_dir, f's{grid_info}o{origin_str}_tetr.xdmf')
    output_current_path = os.path.join(output_dir, f's{grid_info}o{origin_str}_current.xdmf')
    output_potential_path = os.path.join(output_dir, f's{grid_info}o{origin_str}_potential.xdmf')

    with XDMFFile(MPI.COMM_WORLD, tetr_mesh_path, "r") as infile3:
        mesh = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')

    mesh_dim = mesh.topology.dim
    V = FunctionSpace(mesh, ("Lagrange", 2))

    # Dirichlet BCs
    u0 = Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(1)

    u1 = Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0)
    x0facet = locate_entities_boundary(mesh, 0,
                                    lambda x: np.isclose(x[0], 0.0))
    x1facet = locate_entities_boundary(mesh, 0,
                                    lambda x: np.isclose(x[0], Lx))
    x0bc = DirichletBC(u0, locate_dofs_topological(V, 0, x0facet))
    x1bc = DirichletBC(u1, locate_dofs_topological(V, 0, x1facet))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = Constant(mesh, ScalarType(0))
    g = Constant(mesh, ScalarType(0))

    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx(x) + inner(g, v) * ds(mesh)

    problem = LinearProblem(a, L, bcs=[x0bc, x1bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    # When we want to compute the solution to the problem, we can specify
    # what kind of solver we want to use.
    print('solving problem..')
    uh = problem.solve()

    # Save solution in XDMF format
    with XDMFFile(MPI.COMM_WORLD, output_potential_path, "w") as outfile:
        outfile.write_mesh(mesh)
        outfile.write_function(uh)

    # Update ghost entries and plot
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Post-processing: Compute derivatives
    grad_u = grad(uh)

    W = FunctionSpace(mesh, ("Lagrange", 1))
    current_expr = dolfinx.fem.Expression(ufl.sqrt(inner(grad_u, grad_u)), W.element.interpolation_points)
    current_h = Function(W)
    current_h.interpolate(current_expr)

    with XDMFFile(MPI.COMM_WORLD, output_current_path, "w") as file:
        file.write_mesh(mesh)
        file.write_function(current_h)
