#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import math
import os
import timeit

import dolfinx
import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
import pyvista
import pyvista as pv
import pyvistaqt as pvqt
import ufl
import warnings

from dolfinx import cpp, default_real_type, default_scalar_type, fem, io, la, mesh, nls, plot
from dolfinx.fem import petsc
from dolfinx.io import gmshio, VTXWriter
from dolfinx.nls import petsc as petsc_nls
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from IPython.display import Image

from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 dot, div, dx, ds, dS, grad, inner, grad, avg, jump)

import commons, configs, geometry, utils

warnings.simplefilter('ignore')


kappa_elec = 0.1
kappa_pos_am = 0.2
faraday_const = 96485
R = 8.3145
T = 298


def ocv(sod, L=1, k=2):
    return 2.5 + (1/k) * np.log((L - sod) / sod)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Current Collector.')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="lithium_metal_3d_cc_2d")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1.0, type=float)
    parser.add_argument("--Wa", help="Wagna number: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e3, type=float)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='MICRON_TO_METER', type=str)

    args = parser.parse_args()
    start_time = timeit.default_timer()
    Wa = args.Wa
    voltage = args.voltage
    Wa_n = Wa
    Wa_p = Wa
    comm = MPI.COMM_WORLD
    encoding = io.XDMFFile.Encoding.HDF5
    micron = 1e-6
    LX, LY, LZ = [float(vv) * micron for vv in args.dimensions.split("-")]
    workdir = os.path.join(args.mesh_folder, str(Wa_n) + "-" + str(Wa_p))
    utils.make_dir_if_missing(workdir)
    output_meshfile = os.path.join(args.mesh_folder, 'mesh.msh')
    potential_resultsfile = os.path.join(workdir, "potential.bp")
    concentration_resultsfile = os.path.join(workdir, "concentration.bp")
    current_resultsfile = os.path.join(workdir, "current.bp")
    simulation_metafile = os.path.join(workdir, "simulation.json")

    markers = commons.Markers()

    # ### Read input geometry
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = gmshio.read_from_msh(output_meshfile, comm, partitioner=partitioner)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)

    # tag internal facets as 0
    ft_imap = domain.topology.index_map(fdim)
    num_facets = ft_imap.size_local + ft_imap.num_ghosts
    indices = np.arange(0, num_facets)
    values = np.zeros(indices.shape, dtype=np.intc)

    values[ft.indices] = ft.values
    ft = mesh.meshtags(domain, fdim, indices, values)
    ct = mesh.meshtags(domain, tdim, ct.indices, ct.values)
    dx = ufl.Measure("dx", domain=domain, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=ft)

    # ### Function Spaces
    V = fem.FunctionSpace(domain, ("DG", 1))
    W = fem.functionspace(domain, ("DG", 1, (3,)))
    Z = fem.functionspace(domain, ("CG", 1, (3,)))
    Q = fem.FunctionSpace(domain, ("DG", 0))
    u = fem.Function(V, name='potential')
    v = ufl.TestFunction(V)
    current_h = fem.Function(W, name='current_density')
    kappa = fem.Function(Q, name='conductivity')

    n = ufl.FacetNormal(domain)
    x = ufl.SpatialCoordinate(domain)
    h = ufl.CellDiameter(domain)
    h_avg = avg(h)

    cells_elec = ct.find(markers.electrolyte)
    kappa.x.array[cells_elec] = np.full_like(cells_elec, kappa_elec, dtype=default_scalar_type)

    cells_pos_am = ct.find(markers.positive_am)
    kappa.x.array[cells_pos_am] = np.full_like(cells_pos_am, kappa_elec, dtype=default_scalar_type)

    x = SpatialCoordinate(domain)

    f = fem.Constant(domain, PETSc.ScalarType(0))
    g = fem.Constant(domain, PETSc.ScalarType(0))

    u_left = fem.Function(V)
    with u_left.vector.localForm() as u0_loc:
        u0_loc.set(0)
    u_right = fem.Function(V)
    with u_right.vector.localForm() as u1_loc:
        u1_loc.set(voltage)

    i0_n = kappa_elec * R * T / (Wa_n * faraday_const * LX)
    i0_p = kappa_elec * R * T / (Wa_p * faraday_const * LX)

    u_ocv = 0.15
    V_left = 0

    alpha = 15
    gamma = 15
    i_loc = inner((kappa * grad(u))('-'), n("+"))
    u_jump = 2 * ufl.ln(0.5 * i_loc/i0_p + ufl.sqrt((0.5 * i_loc/i0_p)**2 + 1)) * (R * T / faraday_const)

    F = kappa * inner(grad(u), grad(v)) * dx - f * v * dx - kappa * inner(grad(u), n) * v * ds

    # Add DG/IP terms
    F += - avg(kappa) * inner(jump(u, n), avg(grad(v))) * dS(0)
    # F += - inner(jump(kappa * u, n), avg(grad(v))) * dS(0)
    F += - inner(avg(kappa * grad(u)), jump(v, n)) * dS(0)
    # F += + avg(u) * inner(jump(kappa, n), avg(grad(v))) * dS(0)
    F += alpha / h_avg * avg(kappa) * inner(jump(v, n), jump(u, n)) * dS(0)

    # Internal boundary
    F += + avg(kappa) * dot(avg(grad(v)), (u_ocv - u_jump) * n('+')) * dS(markers.electrolyte_v_positive_am)
    F += -alpha / h_avg * avg(kappa) * dot(jump(v, n), (u_ocv - u_jump) * n('+')) * dS(markers.electrolyte_v_positive_am)

    # # Symmetry
    F += - avg(kappa) * inner(jump(u, n), avg(grad(v))) * dS(markers.electrolyte_v_positive_am)

    # # Coercivity
    F += alpha / h_avg * avg(kappa) * inner(jump(u, n), jump(v, n)) * dS(markers.electrolyte_v_positive_am)

    # Nitsche Dirichlet BC terms on left and right boundaries
    # F += - kappa * (u - u_left) * inner(n, grad(v)) * ds(markers.left)
    # F += -gamma / h * (u - u_left) * v * ds(markers.left)
    F += - kappa * (u - u_right) * inner(n, grad(v)) * ds(markers.right) 
    F += -gamma / h * (u - u_right) * v * ds(markers.right)

    # Nitsche Neumann BC terms on insulated boundary
    F += -g * v * ds(markers.insulated_electrolyte) + gamma * h * g * inner(grad(v), n) * ds(markers.insulated_electrolyte)
    F += - gamma * h * inner(inner(grad(u), n), inner(grad(v), n)) * ds(markers.insulated_electrolyte)
    F += -g * v * ds(markers.insulated_positive_am) + gamma * h * g * inner(grad(v), n) * ds(markers.insulated_positive_am)
    F += - gamma * h * inner(inner(grad(u), n), inner(grad(v), n)) * ds(markers.insulated_positive_am)

    # kinetics boundary - neumann
    F += - gamma * h * inner(inner(kappa * grad(u), n), inner(grad(v), n)) * ds(markers.left)
    F -= - gamma * h * 2 * i0_n * ufl.sinh(0.5 * faraday_const / R / T * (V_left - u - 0)) * inner(grad(v), n) * ds(markers.left)

    problem = petsc.NonlinearProblem(F, u)
    solver = petsc_nls.NewtonSolver(comm, problem)
    solver.convergence_criterion = "residual"
    solver.maximum_iterations = 25
    # solver.atol = 1.0e-8
    # solver.rtol = 1.0e2 * np.finfo(default_real_type).eps

    ksp = solver.krylov_solver

    opts = PETSc.Options()
    ksp.setMonitor(lambda _, it, residual: print(it, residual))
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"

    ksp.setFromOptions()
    n_iters, converged = solver.solve(u)
    # u.x.scatter_forward()
    if converged:
        print(f"Converged in {n_iters} iterations")

    current_expr = fem.Expression(-kappa * grad(u), W.element.interpolation_points())
    current_h.interpolate(current_expr)

    current_expr = fem.Expression(-kappa * grad(u), W.element.interpolation_points())
    current_h.interpolate(current_expr)
    current_cg = fem.Function(W, name='current_density')
    current_cg.interpolate(current_h)

    with VTXWriter(comm, potential_resultsfile, [u], engine="BP4") as vtx:
        vtx.write(0.0)

    with VTXWriter(comm, current_resultsfile, [current_h], engine="BP4") as vtx:
        vtx.write(0.0)

    I_neg_charge_xfer = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(current_h, n) * ds(markers.left))), op=MPI.SUM)
    I_pos_charge_xfer = domain.comm.allreduce(fem.assemble_scalar(fem.form(2 * i0_p * ufl.sinh(0.5*faraday_const / R / T * (u("+") - u("-") - u_ocv)) * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)
    I_pos_charge_xfer1 = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(current_h("-"), n("-")) * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)
    I_pos_charge_xfer2 = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(current_cg("-"), n("-")) * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)
    I_right = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(current_h, n) * ds(markers.right))), op=MPI.SUM)
    I_insulated_elec = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(inner(current_h, n)) * ds(markers.insulated_electrolyte))), op=MPI.SUM)
    I_insulated_pos_am = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(inner(current_h, n)) * ds(markers.insulated_positive_am))), op=MPI.SUM)
    I_insulated = I_insulated_elec + I_insulated_pos_am
    area_left = domain.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * ds(markers.left))), op=MPI.SUM)
    area_neg_charge_xfer = domain.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * ds(markers.left))), op=MPI.SUM)
    area_pos_charge_xfer = domain.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)
    area_right = domain.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * ds(markers.right))), op=MPI.SUM)
    i_sup_left = np.abs(I_neg_charge_xfer / area_neg_charge_xfer)
    i_sup = np.abs(I_right / area_right)
    
    eta_p = domain.comm.allreduce(fem.assemble_scalar(fem.form((u("+") - u("-") - u_ocv) * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM) / area_pos_charge_xfer
    u_avg_right = fem.assemble_scalar(fem.form(u * ds(markers.right))) / area_right
    u_avg_left = fem.assemble_scalar(fem.form(u * ds(markers.left))) / area_left
    u_stdev_right = np.sqrt(fem.assemble_scalar(fem.form((u - u_avg_right) ** 2 * ds(markers.right))) / area_right)
    u_stdev_left = np.sqrt(fem.assemble_scalar(fem.form((u - u_avg_left) ** 2 * ds(markers.left))) / area_left)
    eta_n = u_avg_left
    simulation_metadata = {
        "Negative Wagner Number": f"{Wa_n:.1e}",
        "Positive Wagner Number": f"{Wa_p:.1e}",
        "Negative Overpotential [V]": eta_n,
        "Positive Overpotential [V]": eta_p,
        "Voltage": voltage,
        "dimensions": args.dimensions,
        "Superficial current density [A/m2]": f"{np.abs(i_sup):.2e} [A/m2]",
        "Current at negative am - electrolyte boundary": f"{np.abs(I_neg_charge_xfer):.2e} A",
        "Current at electrolyte - positive am boundary": f"{np.abs(I_pos_charge_xfer):.2e} A",
        "Current at right boundary": f"{np.abs(I_right):.2e} A",
        "Current at insulated boundary": f"{I_insulated:.2e} A",
    }
    if comm.rank == 0:
        print(f"Negative Wagner Number: {Wa_n:.1e}")
        print(f"Positive Wagner Number: {Wa_p:.1e}")
        print(2 * i0_p * math.sinh(0.5 * faraday_const / (R * T) * eta_p) * area_pos_charge_xfer, I_pos_charge_xfer1, I_pos_charge_xfer2)
        print(f"Left - avg potential  : {u_avg_left:.3e}, stdev potential  : {u_stdev_left:.3e}")
        print(f"Right - avg potential : {u_avg_right:.3e}, stdev potential  : {u_stdev_right:.3e}")
        print(f"Negative overpotential over Positive overpotential: {eta_n/eta_p:.3f}")
        print(f"Positive area over Negative area: {area_pos_charge_xfer/area_neg_charge_xfer:.3f}")
        print(f"Voltage: {voltage} [V]")
        print(f"Negative Overpotential: {eta_n:.3e} [V]")
        print(f"Positive Overpotential: {eta_p:.3e} [V]")
        print(f"superficial current density @ left: {np.abs(i_sup_left):.2e} [A/m2]")
        print(f"superficial current density @ right: {np.abs(i_sup):.2e} [A/m2]")
        print(f"Current at negative_am - electrolyte boundary: {np.abs(I_neg_charge_xfer):.2e} A")
        print(f"Current at electrolyte - positive am boundary: {np.abs(I_pos_charge_xfer):.2e} A")
        print(f"Current at right boundary: {np.abs(I_right):.2e} A")
        print(f"Current at insulated boundary: {I_insulated:.2e} A")
        print(f"Float precision: {np.finfo(float).eps}")
        with open(simulation_metafile, "w", encoding='utf-8') as f:
            json.dump(simulation_metadata, f, ensure_ascii=False, indent=4)
        print(f"Time elapsed: {int(timeit.default_timer() - start_time):3.5f}s")
