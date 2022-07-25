#!/usr/bin/env python3
# coding: utf-8
import sys

import argparse
import dolfinx
import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc

import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='computes specific area')
    parser.add_argument('--coverage', type=float, help='fraction of lower current collector that is conductive', required=True)
    parser.add_argument('--Lx', help='length', required=True, type=int)
    parser.add_argument('--Ly', help='width', required=True, type=int)
    parser.add_argument("--w", help='slice width along x', nargs='?', const=1, default=10, type=float)
    parser.add_argument("--h", help='slice position along y', nargs='?', const=1, default=0.5, type=float)
    parser.add_argument("--voltage", help='voltage drop (one end held at potential of 0)', nargs='?', const=1, default=1, type=int)

    args = parser.parse_args()
    coverage = np.around(args.coverage, 2)
    Lx = args.Lx
    Ly = args.Ly
    w = args.w / Lx
    h = args.h / Ly
    voltage = args.voltage
    lower_cov = 0.5 * (1 - coverage) * Lx
    upper_cov = Lx - 0.5 * (1 - coverage) * Lx
    meshname = f'current_constriction/{h:.3}_{w:.3}'
    utils.make_dir_if_missing('current_constriction')
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{meshname}.xdmf", "r") as infile3:
            msh = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
            ct = infile3.read_meshtags(msh, name="Grid")

    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim-1)

    Q = dolfinx.fem.FunctionSpace(msh, ("DG", 0))
    kappa = 1.0

    V = dolfinx.fem.FunctionSpace(msh, ("Lagrange", 1))

    # Dirichlet BCs
    u0 = dolfinx.fem.Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(voltage)

    u1 = dolfinx.fem.Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0)
    partially_insulated = lambda x: np.logical_and(np.isclose(x[1], 0.0), np.logical_and(lower_cov <= x[0],  x[0] <= upper_cov))
    x0facet = dolfinx.mesh.locate_entities_boundary(msh, 1, partially_insulated)
    x1facet = dolfinx.mesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[1], Ly))
    x0bc = dolfinx.fem.dirichletbc(u0, dolfinx.fem.locate_dofs_topological(V, 1, x0facet))
    x1bc = dolfinx.fem.dirichletbc(u1, dolfinx.fem.locate_dofs_topological(V, 1, x1facet))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = dolfinx.fem.Constant(msh, PETSc.ScalarType(0))
    g = dolfinx.fem.Constant(msh, PETSc.ScalarType(0))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ufl.ds

    options =  {
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "ksp_atol": 1.0e-12,
                "ksp_rtol": 1.0e-12
                }
    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[x0bc, x1bc], petsc_options=options)
    uh = problem.solve()

    with dolfinx.io.XDMFFile(msh.comm, f"current_constriction/{h:.3}_{w:.3}_{coverage:.2}_{voltage}_potential.xdmf", "w") as file:
        file.write_mesh(msh)
        file.write_function(uh)
    grad_u = ufl.grad(uh)

    W = dolfinx.fem.FunctionSpace(msh, ("Lagrange", 1))

    current_expr = dolfinx.fem.Expression(kappa * ufl.sqrt(ufl.inner(grad_u, grad_u)), W.element.interpolation_points)
    current_h = dolfinx.fem.Function(W)
    current_h.interpolate(current_expr)

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"current_constriction/{h:.3}_{w:.3}_{coverage:.2}_{voltage}_current.xdmf", "w") as file:
        file.write_mesh(msh)
        file.write_function(current_h)

    left_facets = dolfinx.mesh.locate_entities_boundary(msh, msh.topology.dim - 1, lambda x: np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], Ly)))
    left_dofs = dolfinx.fem.locate_dofs_topological(V, msh.topology.dim - 1, left_facets)
    n = -ufl.FacetNormal(msh)  # outward pointing
    dobs = ufl.Measure("ds", domain=msh)
    solution_trace_norm = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(ufl.grad(uh), n) ** 2 * dobs))
    print(f"Homogeneous Neumann BC trace: {solution_trace_norm}")