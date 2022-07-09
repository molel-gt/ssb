#!/usr/bin/env python3

import os
import time

import argparse
import dolfinx
import logging
import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc

import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run simulation..')
    parser.add_argument('--grid_info', help='Nx-Ny-Nz', required=True)
    parser.add_argument('--origin', default=(0, 0, 0), help='where to extract grid from')
    parser.add_argument("--insulated_marker", nargs='?', const=1, default=4228, type=int)

    args = parser.parse_args()
    start = time.time()

    if isinstance(args.origin, str):
        origin = tuple(map(lambda v: int(v), args.origin.split(",")))
    else:
        origin = args.origin
    origin_str = "_".join([str(v) for v in origin])
    grid_info = args.grid_info
    FORMAT = f'%(asctime)s: %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(f'{grid_info} {origin_str}')
    logger.setLevel('INFO')
    Ly = int(grid_info.split("-")[1]) - 1
    working_dir = os.path.abspath(os.path.dirname(__file__))
    meshes_dir = os.path.join(working_dir, 'mesh')
    output_dir = os.path.join(working_dir, 'output')
    utils.make_dir_if_missing(meshes_dir)
    utils.make_dir_if_missing(output_dir)
    tetr_mesh_path = os.path.join(meshes_dir, f's{grid_info}o{origin_str}_tetr.xdmf')
    tria_mesh_path = os.path.join(meshes_dir, f's{grid_info}o{origin_str}_tria.xdmf')
    output_current_path = os.path.join(output_dir, f's{grid_info}o{origin_str}_current.xdmf')
    output_potential_path = os.path.join(output_dir, f's{grid_info}o{origin_str}_potential.xdmf')

    logger.info("Loading mesh..")

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, tetr_mesh_path, "r") as infile3:
        mesh = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
        ct = infile3.read_meshtags(mesh, name="Grid")
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, tria_mesh_path, "r") as infile3:
        mesh_facets = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
        facets_ct = infile3.read_meshtags(mesh, name="Grid")

    logger.info("Loaded mesh.")
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 2))

    # Dirichlet BCs
    u0 = dolfinx.fem.Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(1.0)

    u1 = dolfinx.fem.Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0.0)
    x0facet = dolfinx.mesh.locate_entities_boundary(mesh, 2,
                                    lambda x: np.isclose(x[1], 0.0))
    x1facet = dolfinx.mesh.locate_entities_boundary(mesh, 2,
                                    lambda x: np.isclose(x[1], Ly))
    x0bc = dolfinx.fem.dirichletbc(u0, dolfinx.fem.locate_dofs_topological(V, 2, x0facet))
    x1bc = dolfinx.fem.dirichletbc(u1, dolfinx.fem.locate_dofs_topological(V, 2, x1facet))

    # Neumann BC
    insulated_marker = args.insulated_marker
    insulated_cells = facets_ct.indices[facets_ct.values == insulated_marker]
    insulated_tags = dolfinx.mesh.meshtags(mesh, 2, insulated_cells, insulated_marker)
    n = -ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=insulated_tags, subdomain_id=insulated_marker)
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0))
    g = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0))

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ds

    options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    options = {
               "ksp_type": "gmres",
               "pc_type": "hypre",
               "ksp_rtol": 1.0e-12
               }
    options = {
               "ksp_type": "cg",
               "pc_type": "gamg",
               "ksp_rtol": 1.0e-12,
    }

    model = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[x0bc, x1bc], petsc_options=options)

    logger.info('Solving problem..')
    uh = model.solve()
    logger.info("Writing results to file..")
    
    # Save solution in XDMF format
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, output_potential_path, "w") as outfile:
        outfile.write_mesh(mesh)
        outfile.write_function(uh)

    # # Update ghost entries and plot
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    # Post-processing: Compute derivatives
    grad_u = ufl.grad(uh)

    W = dolfinx.fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
    current_expr = dolfinx.fem.Expression(-grad_u, W.element.interpolation_points)
    current_h = dolfinx.fem.Function(W)
    current_h.interpolate(current_expr)

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, output_current_path, "w") as file:
        file.write_mesh(mesh)
        file.write_function(current_h)
    logger.info("Wrote results to file.")

    solution_trace_norm = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(ufl.grad(uh), n) ** 2 * ds)) ** 0.5
    logger.info(f"Homogeneous Neumann BC trace: {solution_trace_norm}")
    logger.info("Time elapsed: {:,} seconds".format(int(time.time() - start)))