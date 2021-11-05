#!/usr/bin/env python3

import os

import argparse
import dolfinx
import numpy as np
import ufl
from dolfinx import (DirichletBC, Function, FunctionSpace, fem,
                     plot, BoxMesh
                     )
from dolfinx.fem import locate_dofs_topological
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from dolfinx.cpp.mesh import CellType
from mpi4py import MPI
from petsc4py import PETSc
from ufl import cos, ds, dx, exp, grad, inner, pi, sin


def make_dir_if_missing(f_path):
    """"""
    os.makedirs(f_path, exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run simulation..')
    parser.add_argument('--working_dir', help='bmp files parent directory', required=True)
    parser.add_argument('--img_sub_dir', help='bmp files parent directory', required=True)
    parser.add_argument('--grid_info', help='gridSize_startPos_endPos', required=True)
    parser.add_argument('--file_shape', help='shape of image data array', required=True,
                        type=lambda s: [int(item) for item in s.split('_')])

    args = parser.parse_args()
    files_dir = os.path.join(args.working_dir, args.img_sub_dir)
    file_shape = args.file_shape
    grid_info = args.grid_info
    grid_size = int(grid_info.split("_")[0])
    meshes_dir = os.path.join(args.working_dir, 'mesh', grid_info)
    output_dir = os.path.join(args.working_dir, 'output', grid_info)
    make_dir_if_missing(meshes_dir)
    make_dir_if_missing(output_dir)
    tetr_mesh_path = os.path.join(meshes_dir, 'mesh_tetr.xdmf')
    tria_mesh_path = os.path.join(meshes_dir, 'mesh_tria.xdmf')
    output_path = os.path.join(output_dir, 'output.xdmf')

    with XDMFFile(MPI.COMM_WORLD, tetr_mesh_path, "r") as infile3:
        mesh = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.shared_facet, 'Grid')
    print("done loading tetrahedral mesh")

    # with XDMFFile(MPI.COMM_WORLD, tria_mesh_path, "r") as infile2:
    #     mesh_2d = infile2.read_mesh(dolfinx.cpp.mesh.GhostMode.shared_facet, "Grid")
    # print("done reading triangle mesh")

    V = FunctionSpace(mesh, ("Lagrange", 2))

    # Dirichlet BCs
    u0 = Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(1)

    u1 = Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0)
    x0facet = locate_entities_boundary(mesh, 2,
                                    lambda x: np.isclose(x[0], 0.0))
    x1facet = locate_entities_boundary(mesh, 2,
                                    lambda x: np.isclose(x[0], grid_size))
    x0bc = DirichletBC(u0, locate_dofs_topological(V, 2, x0facet))
    x1bc = DirichletBC(u1, locate_dofs_topological(V, 2, x1facet))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = 0
    g = sin(2*pi*x[1]/grid_size) * sin(2*pi*x[2]/grid_size)

    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx(x) + inner(g, v) * ds(mesh)

    print("setting problem..")

    problem = fem.LinearProblem(a, L, bcs=[x0bc, x1bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    # When we want to compute the solution to the problem, we can specify
    # what kind of solver we want to use.
    print('solving problem..')
    uh = problem.solve()

    # Save solution in XDMF format
    with XDMFFile(MPI.COMM_WORLD, output_path, "w") as outfile:
        outfile.write_mesh(mesh)
        outfile.write_function(uh)

    # Update ghost entries and plot
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
