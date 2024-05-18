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
    parser.add_argument("--Wa_n", help="Wagna number for negative electrode: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e-3, type=float)
    parser.add_argument("--Wa_p", help="Wagna number for positive electrode: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e3, type=float)
    parser.add_argument("--gamma", help="interior penalty parameter", nargs='?', const=1, default=15, type=float)
    parser.add_argument("--atol", help="solver absolute tolerance", nargs='?', const=1, default=1e-12, type=float)
    parser.add_argument("--rtol", help="solver relative tolerance", nargs='?', const=1, default=1e-9, type=float)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='MICRON_TO_METER', type=str)

    args = parser.parse_args()
    start_time = timeit.default_timer()
    voltage = args.voltage
    Wa_n = args.Wa_n
    Wa_p = args.Wa_p
    comm = MPI.COMM_WORLD
    encoding = io.XDMFFile.Encoding.HDF5
    micron = 1e-6
    LX, LY, LZ = [float(vv) * micron for vv in args.dimensions.split("-")]
    workdir = os.path.join(args.mesh_folder, str(Wa_n) + "-" + str(Wa_p), str(args.gamma))
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

    dx = ufl.Measure("dx", domain=domain, subdomain_data=ct, metadata={"quadrature_degree": 4})
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft, metadata={"quadrature_degree": 4})

    f_to_c = domain.topology.connectivity(fdim, tdim)
    c_to_f = domain.topology.connectivity(tdim, fdim)
    charge_xfer_facets = ft.find(markers.electrolyte_v_positive_am)
    other_internal_facets = ft.find(0)
    int_facet_domain = []
    for f in charge_xfer_facets:
        if f >= ft_imap.size_local or len(f_to_c.links(f)) != 2:
            continue
        c_0, c_1 = f_to_c.links(f)[0], f_to_c.links(f)[1]
        subdomain_0, subdomain_1 = ct.values[[c_0, c_1]]
        local_f_0 = np.where(c_to_f.links(c_0) == f)[0][0]
        local_f_1 = np.where(c_to_f.links(c_1) == f)[0][0]
        if subdomain_0 > subdomain_1:
            int_facet_domain.append(c_0)
            int_facet_domain.append(local_f_0)
            int_facet_domain.append(c_1)
            int_facet_domain.append(local_f_1)
        else:
            int_facet_domain.append(c_1)
            int_facet_domain.append(local_f_1)
            int_facet_domain.append(c_0)
            int_facet_domain.append(local_f_0)

    other_internal_facet_domains = []
    for f in other_internal_facets:
        if f >= ft_imap.size_local or len(f_to_c.links(f)) != 2:
            continue
        c_0, c_1 = f_to_c.links(f)[0], f_to_c.links(f)[1]
        subdomain_0, subdomain_1 = ct.values[[c_0, c_1]]
        local_f_0 = np.where(c_to_f.links(c_0) == f)[0][0]
        local_f_1 = np.where(c_to_f.links(c_1) == f)[0][0]
        other_internal_facet_domains.append(c_0)
        other_internal_facet_domains.append(local_f_0)
        other_internal_facet_domains.append(c_1)
        other_internal_facet_domains.append(local_f_1)
    int_facet_domains = [(markers.electrolyte_v_positive_am, int_facet_domain)]#, (0, other_internal_facet_domains)]

    dS = ufl.Measure("dS", domain=domain, subdomain_data=int_facet_domains)

    # ### Function Spaces
    V = fem.functionspace(domain, ("DG", 1))
    W = fem.functionspace(domain, ("DG", 1, (3,)))
    Z = fem.functionspace(domain, ("CG", 1, (3,)))
    Q = fem.functionspace(domain, ("DG", 0))
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

    alpha = args.gamma
    gamma = args.gamma
    i_loc = -inner((kappa * grad(u))('+'), n("+"))
    u_jump = 2 * ufl.ln(0.5 * i_loc/i0_p + ufl.sqrt((0.5 * i_loc/i0_p)**2 + 1)) * (R * T / faraday_const)

    F = kappa * inner(grad(u), grad(v)) * dx - f * v * dx - kappa * inner(grad(u), n) * v * ds

    # Add DG/IP terms
    F += - avg(kappa) * inner(jump(u, n), avg(grad(v))) * dS#(0)
    # F += - inner(utils.jump(kappa * u, n), avg(grad(v))) * dS(0)
    F += - inner(avg(kappa * grad(u)), jump(v, n)) * dS#(0)
    # F += + avg(u) * inner(utils.jump(kappa, n), avg(grad(v))) * dS(0)
    F += alpha / h_avg * avg(kappa) * inner(jump(v, n), jump(u, n)) * dS#(0)

    # Internal boundary
    F += + avg(kappa) * dot(avg(grad(v)), (u_jump + u_ocv) * n('+')) * dS(markers.electrolyte_v_positive_am)
    F += -alpha / h_avg * avg(kappa) * dot(jump(v, n), (u_jump + u_ocv) * n('+')) * dS(markers.electrolyte_v_positive_am)

    # # Symmetry
    F += - avg(kappa) * inner(jump(u, n), avg(grad(v))) * dS(markers.electrolyte_v_positive_am)

    # # Coercivity
    F += alpha / h_avg * avg(kappa) * inner(jump(u, n), jump(v, n)) * dS(markers.electrolyte_v_positive_am)

    # Nitsche Dirichlet BC terms on left and right boundaries
    F += - kappa * (u - u_left) * inner(n, grad(v)) * ds(markers.left)
    F += -gamma / h * (u - u_left) * v * ds(markers.left)
    F += - kappa * (u - u_right) * inner(n, grad(v)) * ds(markers.right) 
    F += -gamma / h * (u - u_right) * v * ds(markers.right)

    # Nitsche Neumann BC terms on insulated boundary
    F += -g * v * ds(markers.insulated_electrolyte) + gamma * h * g * inner(grad(v), n) * ds(markers.insulated_electrolyte)
    F += - gamma * h * inner(inner(grad(u), n), inner(grad(v), n)) * ds(markers.insulated_electrolyte)
    F += -g * v * ds(markers.insulated_positive_am) + gamma * h * g * inner(grad(v), n) * ds(markers.insulated_positive_am)
    F += - gamma * h * inner(inner(grad(u), n), inner(grad(v), n)) * ds(markers.insulated_positive_am)

    # kinetics boundary - neumann
    # F += - gamma * h * inner(inner(kappa * grad(u), n), inner(grad(v), n)) * ds(markers.left)
    # F -= - gamma * h * 2 * i0_n * ufl.sinh(0.5 * faraday_const / R / T * (V_left - u - 0)) * inner(grad(v), n) * ds(markers.left)

    problem = petsc.NonlinearProblem(F, u)
    solver = petsc_nls.NewtonSolver(comm, problem)
    solver.convergence_criterion = "residual"
    # solver.maximum_iterations = 25
    solver.rtol = args.rtol
    solver.atol = args.atol

    ksp = solver.krylov_solver

    opts = PETSc.Options()
    # ksp.setMonitor(lambda _, it, residual: print(it, residual))
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "gmres"
    # opts[f"{option_prefix}pc_type"] = "hypre"

    ksp.setFromOptions()
    n_iters, converged = solver.solve(u)
    u.x.scatter_forward()
    if converged:
        print(f"Converged in {n_iters} iterations")

    current_expr = fem.Expression(-kappa * grad(u), W.element.interpolation_points())
    current_h.interpolate(current_expr)
    current_cg = fem.Function(W, name='current_density')
    current_cg.interpolate(current_h)

    with VTXWriter(comm, potential_resultsfile, [u], engine="BP4") as vtx:
        vtx.write(0.0)

    with VTXWriter(comm, current_resultsfile, [current_h], engine="BP4") as vtx:
        vtx.write(0.0)

    I_neg_charge_xfer = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(current_h, n) * ds(markers.left))), op=MPI.SUM)
    I_pos_am = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(current_h("+"), n("+")) * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)
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
    i_pos_am = I_pos_am / area_pos_charge_xfer
    std_dev_i_pos_am = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((inner(current_h("+"), n("+")) - i_pos_am) ** 2 * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)/area_pos_charge_xfer)
    std_dev_i_pos_am_norm = np.abs(std_dev_i_pos_am / i_pos_am)
    eta_p = domain.comm.allreduce(fem.assemble_scalar(fem.form(2 * R * T / (faraday_const) * utils.arcsinh(0.5 * np.abs(-inner(kappa * grad(u), n)("+"))) * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)
    u_avg_right = domain.comm.allreduce(fem.assemble_scalar(fem.form(u * ds(markers.right))) / area_right, op=MPI.SUM)
    u_avg_left = domain.comm.allreduce(fem.assemble_scalar(fem.form(u * ds(markers.left))) / area_left, op=MPI.SUM)
    u_stdev_right = domain.comm.allreduce(np.sqrt(fem.assemble_scalar(fem.form((u - u_avg_right) ** 2 * ds(markers.right))) / area_right), op=MPI.SUM)
    u_stdev_left = domain.comm.allreduce(np.sqrt(fem.assemble_scalar(fem.form((u - u_avg_left) ** 2 * ds(markers.left))) / area_left), op=MPI.SUM)
    eta_n = u_avg_left
    simulation_metadata = {
        "Negative Wagner Number": f"{Wa_n:.1e}",
        "Positive Wagner Number": f"{Wa_p:.1e}",
        "Negative Overpotential [V]": f"{eta_n:.2e}",
        "Positive Overpotential [V]": f"{eta_p:.2e}",
        "Voltage": voltage,
        "dimensions": args.dimensions,
        "interior penalty (gamma)": args.gamma,
        "average potential left [V]": f"{u_avg_left:.2e}",
        "stdev potential left [V]": f"{u_stdev_left:.2e}",
        "average potential right [V]": f"{u_avg_right:.2e}",
        "stdev potential right [V]": f"{u_stdev_right:.2e}",
        "Superficial current density [A/m2]": f"{np.abs(i_sup):.2e} [A/m2]",
        "Current at negative am - electrolyte boundary": f"{np.abs(I_neg_charge_xfer):.2e} A",
        "Current at electrolyte - positive am boundary": f"{np.abs(I_pos_am):.2e} A",
        "Current at right boundary": f"{np.abs(I_right):.2e} A",
        "Current at insulated boundary": f"{I_insulated:.2e} A",
        "stdev i positive charge transfer": f"{std_dev_i_pos_am:.2e} A/m2",
        "stdev i positive charge transfer (normalized)": f"{std_dev_i_pos_am_norm:.2e}",
        "solver atol": args.atol,
        "solver rtol": args.rtol,

    }
    if comm.rank == 0:
        utils.print_dict(simulation_metadata, padding=50)
        with open(simulation_metafile, "w", encoding='utf-8') as f:
            json.dump(simulation_metadata, f, ensure_ascii=False, indent=4)
        print(f"Time elapsed: {int(timeit.default_timer() - start_time):3.5f}s")
