#!/usr/bin/env python3
# coding: utf-8
import sys
import time

import argparse
import dolfinx
import logging
import numpy as np
import ufl

from dolfinx import cpp, fem, io, mesh, nls, plot
from mpi4py import MPI
from petsc4py import PETSc

import commons, utils

positions = ['left', 'mid', 'right']
KAPPA = 0.1  # 0.1 S/m == mS/cm
markers = commons.SurfaceMarkers()

def insulator_moved_around(x, coverage, Lx, Ly, pos='mid', xlim=[0.125, 0.875], n_pieces=0):
        if n_pieces > 1 and pos != 'mid':
            raise ValueError("Multiple pieces not implemented for piece not centered in the middle")
        if pos == "left":
            lower_cov = 0.5 * (1 - coverage) * Lx - xlim[0] * Lx
            upper_cov = Lx - 0.5 * (1 - coverage) * Lx - xlim[0] * Lx
        if pos == 'mid':
            lower_cov = 0.5 * (1 - coverage) * Lx
            upper_cov = Lx - 0.5 * (1 - coverage) * Lx
        if pos == 'right':
            lower_cov = 0.5 * (1 - coverage) * Lx + xlim[0] * Lx
            upper_cov = Lx - 0.5 * (1 - coverage) * Lx + xlim[0] * Lx
        if n_pieces == 1:
            return lambda x: np.logical_and(np.isclose(x[1], 0.0), np.logical_and(
                np.greater_equal(x[0], lower_cov),  np.greater_equal(upper_cov, x[0])
                )
            )
        else:
            dx = Lx * (coverage / n_pieces)
            space = Lx * ((xlim[1] - xlim[0]) - coverage) / (n_pieces - 1)
            intervals = []
            for i in range(n_pieces + 1):
                if i == 0:
                    intervals.append((xlim[0] * Lx, dx + xlim[0] * Lx))
                else:
                    intervals.append(((dx + space) * (i - 1) + xlim[0] * Lx, dx * i + space * (i -1) + xlim[0] * Lx))
            def fun(x, intervals):
                n = len(intervals)
                if n == 1:
                    return np.logical_and(np.greater_equal(x[0], intervals[0][0]),  np.greater_equal(intervals[0][1], x[0]))
                else:
                    return np.logical_or(
                        np.logical_and(np.greater_equal(x[0], intervals[0][0]),  np.greater_equal(intervals[0][1], x[0])),
                        fun(x, intervals[1:])
                        )
            return lambda x: np.logical_and(np.isclose(x[1], 0.0), fun(x, intervals))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='computes specific area')
    parser.add_argument('--eps', type=float, help='fraction of lower current collector that is conductive', required=True)
    parser.add_argument('--Lx', help='length', required=True, type=int)
    parser.add_argument('--Ly', help='width', required=True, type=int)
    parser.add_argument("--w", help='slice width along x', nargs='?', const=1, default=0, type=float)
    parser.add_argument("--h", help='slice position along y', nargs='?', const=1, default=0, type=float)
    parser.add_argument("--voltage", help='voltage drop (one end held at potential of 0)', nargs='?', const=1, default=1, type=int)
    parser.add_argument("--pos", help='insulator position along x', nargs='?', const=1, default='mid')
    parser.add_argument("--n_pieces", help='insulator position along x', nargs='?', const=1, default=1, type=int)

    args = parser.parse_args()
    start = time.time()
    FORMAT = f'%(asctime)s: %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(f'current_constriction:{args.n_pieces}:{args.eps}:{args.voltage}')
    logger.setLevel('INFO')
    pos = args.pos
    n_pieces = int(args.n_pieces)
    eps = np.around(args.eps, 4)
    Lx = args.Lx
    Ly = args.Ly
    w = args.w / Lx
    h = args.h / Ly
    voltage = args.voltage
    lower_cov = 0.5 * Lx - 0.5 * eps * Lx
    upper_cov = 0.5 * Lx + 0.5 * eps * Lx
    tria_meshname = f'current_constriction/{h:.3}_{w:.3}_pos-{pos}_pieces-{n_pieces}_{eps}_tria'
    line_meshname = f'current_constriction/{h:.3}_{w:.3}_pos-{pos}_pieces-{n_pieces}_{eps}_line'
    utils.make_dir_if_missing('current_constriction')
    with io.XDMFFile(MPI.COMM_WORLD, f"{tria_meshname}.xdmf", "r") as infile3:
            msh = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
            ct = infile3.read_meshtags(msh, name="Grid")
    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)
    with io.XDMFFile(MPI.COMM_WORLD, f"{line_meshname}.xdmf", "r") as infile3:
        ft = infile3.read_meshtags(msh, name="Grid")

    # left_cc_marker, right_cc_marker, insulated_marker = sorted([int(v) for v in set(facets_ct.values)])

    Q = fem.FunctionSpace(msh, ("DG", 0))

    V = fem.FunctionSpace(msh, ("Lagrange", 1))
    meshtags = mesh.meshtags(msh, msh.topology.dim - 1, ft.indices, ft.values)
    n = -ufl.FacetNormal(msh)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=meshtags)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = fem.Constant(msh, PETSc.ScalarType(0))
    g = fem.Constant(msh, PETSc.ScalarType(0))

    # Dirichlet BCs
    u0 = fem.Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(voltage)

    u1 = fem.Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0)

    # partially_insulated = insulator_moved_around(x, eps, Lx, Ly, n_pieces=n_pieces, pos=pos)
    x0facet = ft.find(markers.left_cc)
    x1facet = ft.find(markers.right_cc)
    x0bc = fem.dirichletbc(u0, fem.locate_dofs_topological(V, 1, x0facet))
    x1bc = fem.dirichletbc(u1, fem.locate_dofs_topological(V, 1, x1facet))

    a = ufl.inner(KAPPA * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ufl.ds(markers.insulated)

    # options = {
    #             "ksp_type": "gmres",
    #             "pc_type": "hypre",
    #             # "ksp_atol": 1.0e-12,
    #             "ksp_rtol": 1.0e-16
    #             }
    options = {"ksp_type": "preonly", "pc_type": "lu"}
    problem = fem.petsc.LinearProblem(a, L, bcs=[x0bc, x1bc], petsc_options=options)
    uh = problem.solve()

    with io.XDMFFile(msh.comm, f"current_constriction/{h:.3}_{w:.3}_{eps:.4}_{voltage}_pos-{pos}_pieces-{n_pieces}_potential.xdmf", "w") as file:
        file.write_mesh(msh)
        file.write_function(uh)
    grad_u = ufl.grad(uh)

    W = fem.VectorFunctionSpace(msh, ("Lagrange", 1))

    current_expr = fem.Expression(-KAPPA * grad_u, W.element.interpolation_points())
    current_h = fem.Function(W)
    current_h.interpolate(current_expr)

    with io.XDMFFile(MPI.COMM_WORLD, f"current_constriction/{h:.3}_{w:.3}_{eps:.4}_{voltage}_pos-{pos}_pieces-{n_pieces}_current.xdmf", "w") as file:
        file.write_mesh(msh)
        file.write_function(current_h)

    insulated_area = fem.assemble_scalar(fem.form(1 * ds(markers.insulated)))
    area_left_cc = fem.assemble_scalar(fem.form(1 * ds(markers.left_cc)))
    area_right_cc = fem.assemble_scalar(fem.form(1 * ds(markers.right_cc)))
    total_area = area_left_cc + area_right_cc + insulated_area
    I_left_cc = fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.left_cc)))
    I_right_cc = fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.right_cc)))
    I_insulated = fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.insulated)))
    total_volume = fem.assemble_scalar(fem.form(1 * ufl.dx(msh)))
    solution_trace_norm = I_insulated
    avg_solution_trace_norm = solution_trace_norm / insulated_area
    error = 100 * 2 * abs(abs(I_left_cc) - abs(I_right_cc)) / (abs(I_left_cc) + abs(I_right_cc))
    logger.info("**************************RESULTS-SUMMARY******************************************")
    logger.info("Contact Area @ left cc                          : {:.1f}".format(area_left_cc))
    logger.info("Contact Area @ right cc                         : {:.1f}".format(area_right_cc))
    logger.info("Current density @ left cc                       : {:.6f}".format(I_left_cc / area_left_cc))
    logger.info("Current density @ right cc                      : {:.6f}".format(I_right_cc / area_right_cc))
    logger.info("Insulated Area                                  : {:.1f}".format(insulated_area))
    logger.info("Total Area                                      : {:.1f}".format(total_area))
    logger.info("Total Volume                                    : {:,}".format(int(total_volume)))
    logger.info("Electrolyte Volume Fraction                     : {:0.4f}".format(total_volume/(Lx * Ly * 1)))
    logger.info(f"Bulk conductivity [S.m-1]                       : {KAPPA:.4f}")
    logger.info("Effective conductivity [S.m-1]                  : {:.4f}".format(Ly * abs(I_left_cc) / (voltage * (Lx * 1))))
    logger.info(f"Homogeneous Neumann BC trace                    : {solution_trace_norm:.2e}")
    logger.info(f"Area-averaged Homogeneous Neumann BC trace      : {avg_solution_trace_norm:.2e}")
    logger.info(f"Deviation in current at two current collectors  : {error:.2f}%")
    logger.info(f"Time elapsed                                    : {int(time.time() - start):,} seconds")
    logger.info("*************************END-OF-SUMMARY*******************************************")
