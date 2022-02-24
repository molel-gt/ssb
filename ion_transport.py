#!/usr/bin/env python3

import os

import argparse
import dolfinx
import numpy as np
import ufl
from dolfinx.fem import (dirichletbc, Function, FunctionSpace, VectorFunctionSpace)
from dolfinx.fem import locate_dofs_topological, LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
from petsc4py import PETSc
from ufl import ds, dx, grad, inner


def make_dir_if_missing(f_path):
    """"""
    os.makedirs(f_path, exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run simulation..')
    parser.add_argument('--working_dir', help='bmp files parent directory', required=True)
    parser.add_argument('--grid_info', help='gridSize_startPos_endPos', required=True)
    parser.add_argument('--file_shape', help='shape of image data array', required=True,
                        type=lambda s: [int(item) for item in s.split('_')])

    args = parser.parse_args()
    file_shape = args.file_shape
    grid_info = args.grid_info
    grid_size = int(grid_info.split(".")[0])
    meshes_dir = os.path.join(args.working_dir, 'mesh', grid_info)
    output_dir = os.path.join(args.working_dir, 'output', grid_info)
    make_dir_if_missing(meshes_dir)
    make_dir_if_missing(output_dir)
    tetr_mesh_path = os.path.join(meshes_dir, 'mesh_tetr.xdmf')
    tria_mesh_path = os.path.join(meshes_dir, 'mesh_tria.xdmf')
    output_path = os.path.join(output_dir, 'output.xdmf')

    with XDMFFile(MPI.COMM_WORLD, tetr_mesh_path, "r") as infile3:
        mesh = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
    print("done loading tetrahedral mesh")
    mesh_dim = mesh.topology.dim
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))

    # Dirichlet BCs
    u0 = Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(1)

    u1 = Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0)
    x0facet = locate_entities_boundary(mesh, mesh_dim-1,
                                    lambda x: np.isclose(x[0], 0.0))
    x1facet = locate_entities_boundary(mesh, mesh_dim-1,
                                    lambda x: np.isclose(x[0], grid_size))
    x0bc = dirichletbc(u0, locate_dofs_topological(V, mesh_dim-1, x0facet))
    x1bc = dirichletbc(u1, locate_dofs_topological(V, mesh_dim-1, x1facet))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0.0, 0.0, 0.0)))
    g = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0.0, 0.0, 0.0)))

    a = dolfinx.fem.form(inner(grad(u), grad(v)) * dx(x))
    L = dolfinx.fem.form(inner(f, v) * dx + inner(g, v) * ds)

    A = dolfinx.fem.assemble_matrix(a, bcs=[x0bc, x1bc])
    A.assemble()

    b = dolfinx.fem.assemble_vector(L)
    dolfinx.fem.apply_lifting(b, [a], bcs=[[x0bc, x1bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, [x0bc, x1bc])

    # Set solver options
    opts = PETSc.Options()
    opts["ksp_type"] = "cg"
    opts["ksp_rtol"] = 1.0e-10
    opts["pc_type"] = "gamg"

    # # Use Chebyshev smoothing for multigrid
    opts["mg_levels_ksp_type"] = "chebyshev"
    opts["mg_levels_pc_type"] = "jacobi"

    # # Improve estimate of eigenvalues for Chebyshev smoothing
    opts["mg_levels_esteig_ksp_type"] = "cg"
    opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

    # Create PETSc Krylov solver and turn convergence monitoring on
    solver = PETSc.KSP().create(mesh.comm)
    solver.setFromOptions()

    # Set matrix operator
    solver.setOperators(A)

    uh = Function(V)

    # Set a monitor, solve linear system, and dispay the solver configuration
    # solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
    solver.solve(b, uh.vector)
    # solver.view()

    uh.x.scatter_forward()

    # a = inner(grad(u), grad(v)) * dx
    # L = inner(f, v) * dx(x) + inner(g, v) * ds(mesh)

    # print("setting problem..")

    # problem = LinearProblem(a, L, bcs=[x0bc, x1bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    # # When we want to compute the solution to the problem, we can specify
    # # what kind of solver we want to use.
    # print('solving problem..')
    # uh = problem.solve()

    # Save solution in XDMF format
    with XDMFFile(MPI.COMM_WORLD, output_path, "w") as outfile:
        outfile.write_mesh(mesh)
        outfile.write_function(uh)

    # Update ghost entries and plot
    # uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Post-processing: Compute derivatives
    grad_u = ufl.sym(grad(uh)) #* ufl.Identity(len(uh))

    W = FunctionSpace(mesh, ("Discontinuous Lagrange", 0))
    current_expr = dolfinx.fem.Expression(ufl.sqrt(inner(grad_u, grad_u)), W.element.interpolation_points)
    current_h = Function(W)
    current_h.interpolate(current_expr)

    with XDMFFile(MPI.COMM_WORLD, "current.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(current_h)
