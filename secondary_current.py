#!/usr/bin/env python
# coding: utf-8

import argparse
import itertools
import json
import math
import os
import timeit

import dolfinx
import gmsh
import h5py
import matplotlib.pyplot as plt
import meshio
import numpy as np
import subprocess
import ufl
import warnings

from dolfinx import cpp, default_real_type, default_scalar_type, fem, io, la, log, mesh, nls, plot
from dolfinx.fem import petsc
from dolfinx.io import gmshio, VTXWriter, XDMFFile
from dolfinx.nls import petsc as petsc_nls
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from IPython.display import Image

from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 dot, div, dx, ds, dS, grad, inner, grad, avg, jump)

import commons, configs, geometry, utils

warnings.simplefilter('ignore')


# kappa_elec = 0.1
kappa_pos_am = 0.1
faraday_const = 96485
R = 8.3145
T = 298
kinetics = ("linear", "tafel", "butler_volmer")


def check_kinetics(val):
    if val not in kinetics:
        raise ValueError(f"Invalid kinetics, allowed values are {kinetics.__repr__()}")
    return True


def ocv(sod, L=1, k=2):
    return 2.5 + (1/k) * np.log((L - sod) / sod)


def read_node_ids_for_marker(h5_file_obj, marker):
    line_ids = np.where(np.asarray(h5_file_obj['data2']) == marker)[0]
    lines = np.asarray(h5_file_obj['data1'])[line_ids, :]
    nodes = sorted(list(set(list(itertools.chain.from_iterable(lines.reshape(-1, 1).tolist())))))

    return nodes


def get_surface_overpotential(kappa, i0, u, kinetics_type="butler_volmer"):
    i_loc = -inner((kappa * grad(u))('+'), n("+"))
    if kinetics_type == "butler_volmer":
        return 2 * ufl.ln(0.5 * i_loc/i0 + ufl.sqrt((0.5 * i_loc/i0)**2 + 1)) * (R * T / faraday_const)
    elif kinetics_type == "linear":
        return R * T * i_loc / (i0 * faraday_const)
    elif kinetics_type == "tafel":
        return ufl.sign(i_loc) * R * T / (0.5 * faraday_const) * ufl.ln(np.abs(i_loc)/i_0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='secondary current distribution')
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1.0, type=float)
    parser.add_argument("--u_ocv", help="open-circuit potential", nargs='?', const=1, default=0, type=float)
    parser.add_argument("--Wa_n", help="Wagna number for negative electrode: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e-3, type=float)
    parser.add_argument("--Wa_p", help="Wagna number for positive electrode: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e3, type=float)
    parser.add_argument("--kr", help="ratio of ionic to electronic conductivity", nargs='?', const=1, default=1, type=float)
    parser.add_argument("--gamma", help="interior penalty parameter", nargs='?', const=1, default=15, type=float)
    parser.add_argument("--atol", help="solver absolute tolerance", nargs='?', const=1, default=1e-12, type=float)
    parser.add_argument("--rtol", help="solver relative tolerance", nargs='?', const=1, default=1e-9, type=float)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='MICRON_TO_METER', type=str)
    parser.add_argument('--kinetics', help='kinetics type', nargs='?', const=1, default='butler_volmer', type=str, choices=kinetics)
    parser.add_argument("--plot", help="whether to plot results", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    start_time = timeit.default_timer()
    voltage = args.voltage
    Wa_n = args.Wa_n
    Wa_p = args.Wa_p
    comm = MPI.COMM_WORLD
    encoding = io.XDMFFile.Encoding.HDF5
    micron = 1e-6
    dimensions = utils.extract_dimensions_from_meshfolder(args.mesh_folder)
    LX, LY, LZ = [float(vv) * micron for vv in dimensions.split("-")]
    characteristic_length = min([val for val in [LX, LY, LZ] if val > 0])
    workdir = os.path.join(args.mesh_folder, args.kinetics, str(Wa_n) + "-" + str(Wa_p) + "-" + str(args.kr), str(args.gamma))
    utils.make_dir_if_missing(workdir)
    output_meshfile = os.path.join(args.mesh_folder, 'mesh.msh')
    lines_h5file = os.path.join(args.mesh_folder, 'lines.h5')
    potential_resultsfile = os.path.join(workdir, "potential.bp")
    concentration_resultsfile = os.path.join(workdir, "concentration.bp")
    current_dist_file = os.path.join(workdir, f"current-y-positions-{str(args.Wa_p)}-{str(args.kr)}.png")
    reaction_dist_file = os.path.join(workdir, f"reaction-dist-{str(args.Wa_p)}-{str(args.kr)}.png")
    current_resultsfile = os.path.join(workdir, "current.bp")
    simulation_metafile = os.path.join(workdir, "simulation.json")

    markers = commons.Markers()

    log.set_log_level(log.LogLevel.WARNING)

    # ### Read input geometry
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = gmshio.read_from_msh(output_meshfile, comm, partitioner=partitioner)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)
    x = SpatialCoordinate(domain)

    # tag internal facets as 0
    ft_imap = domain.topology.index_map(fdim)
    num_facets = ft_imap.size_local + ft_imap.num_ghosts
    indices = np.arange(0, num_facets)
    values = np.zeros(indices.shape, dtype=np.intc)

    values[ft.indices] = ft.values


    ft = mesh.meshtags(domain, fdim, indices, values)
    ct = mesh.meshtags(domain, tdim, ct.indices, ct.values)

    print("Finished reading mesh and setting cell and facet tags")

    dx = ufl.Measure("dx", domain=domain, subdomain_data=ct, metadata={"quadrature_degree": 4})
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft, metadata={"quadrature_degree": 4})

    f_to_c = domain.topology.connectivity(fdim, tdim)
    c_to_f = domain.topology.connectivity(tdim, fdim)
    charge_xfer_facets = ft.find(markers.electrolyte_v_positive_am)

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

    # other_internal_facets = np.hstack((ft.find(0), ft.find(markers.left), ft.find(markers.right), ft.find(markers.insulated)))
    # other_internal_facet_domains = []
    # for f in other_internal_facets:
    #     # if f >= ft_imap.size_local or len(f_to_c.links(f)) != 2:
    #     #     continue
    #     if f >= ft_imap.size_local:
    #         continue
    #     else:
    #          if len(f_to_c.links(f)) != 2:
    #             c_0 = f_to_c.links(f)[0]
    #             local_f_0 = np.where(c_to_f.links(c_0) == f)[0][0]
    #             other_internal_facet_domains.append(c_0)
    #             other_internal_facet_domains.append(local_f_0)
    #             continue
    #     c_0, c_1 = f_to_c.links(f)[0], f_to_c.links(f)[1]
    #     subdomain_0, subdomain_1 = ct.values[[c_0, c_1]]
    #     local_f_0 = np.where(c_to_f.links(c_0) == f)[0][0]
    #     local_f_1 = np.where(c_to_f.links(c_1) == f)[0][0]
    #     other_internal_facet_domains.append(c_0)
    #     other_internal_facet_domains.append(local_f_0)
    #     other_internal_facet_domains.append(c_1)
    #     other_internal_facet_domains.append(local_f_1)
    int_facet_domains = [(markers.electrolyte_v_positive_am, int_facet_domain)]  #, (0, other_internal_facet_domains)]

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
    kappa_elec = args.kr * kappa_pos_am
    kappa.x.array[cells_elec] = np.full_like(cells_elec, kappa_elec, dtype=default_scalar_type)

    # kappa_pos_am = kappa_elec/args.kr
    cells_pos_am = ct.find(markers.positive_am)
    kappa.x.array[cells_pos_am] = np.full_like(cells_pos_am, kappa_pos_am, dtype=default_scalar_type)

    f = fem.Constant(domain, PETSc.ScalarType(0))
    g = fem.Constant(domain, PETSc.ScalarType(0))

    u_left = fem.Function(V)
    with u_left.vector.localForm() as u0_loc:
        u0_loc.set(0)
    u_right = fem.Function(V)
    with u_right.vector.localForm() as u1_loc:
        u1_loc.set(voltage)

    i0_n = kappa_elec * R * T / (Wa_n * faraday_const * characteristic_length)
    i0_p = kappa_elec * R * T / (Wa_p * faraday_const * characteristic_length)

    u_ocv = args.u_ocv
    V_left = 0

    alpha = 100#args.gamma
    gamma = 100#args.gamma
    i_loc = -inner((kappa * grad(u))('+'), n("+"))
    u_jump = get_surface_overpotential(kappa, i0_p, u) #2 * ufl.ln(0.5 * i_loc/i0_p + ufl.sqrt((0.5 * i_loc/i0_p)**2 + 1)) * (R * T / faraday_const)

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
    solver.maximum_iterations = 100
    solver.rtol = args.rtol
    solver.atol = args.atol

    ksp = solver.krylov_solver

    opts = PETSc.Options()
    ksp.setMonitor(lambda _, it, residual: print(it, residual))
    ksp.setErrorIfNotConverged(True)
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "cg"
    opts[f"{option_prefix}pc_type"] = "lu"

    ksp.setFromOptions()
    print(f"Secondary current distribution: Wa = {args.Wa_p}, kappa/sigma = {args.kr}")
    n_iters, converged = solver.solve(u)
    u.x.scatter_forward()
    if converged:
        print(f"Converged in {n_iters} iterations")

    current_expr = fem.Expression(-kappa * grad(u), W.element.interpolation_points())
    current_h.interpolate(current_expr)

    with VTXWriter(comm, potential_resultsfile, [u], engine="BP5") as vtx:
        vtx.write(0.0)

    with VTXWriter(comm, current_resultsfile, [current_h], engine="BP5") as vtx:
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
    std_dev_i_pos_am = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((np.abs(inner(current_h("+"), n("+"))) - np.abs(i_pos_am)) ** 2 * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)/area_pos_charge_xfer)
    std_dev_i_pos_am_norm = np.abs(std_dev_i_pos_am / i_pos_am)
    eta_p = domain.comm.allreduce(fem.assemble_scalar(fem.form(2 * R * T / (faraday_const) * utils.arcsinh(0.5 * np.abs(-inner(kappa * grad(u), n)("+"))) * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)
    u_avg_right = domain.comm.allreduce(fem.assemble_scalar(fem.form(u * ds(markers.right))) / area_right, op=MPI.SUM)
    u_avg_left = domain.comm.allreduce(fem.assemble_scalar(fem.form(u * ds(markers.left))) / area_left, op=MPI.SUM)
    u_stdev_right = domain.comm.allreduce(np.sqrt(fem.assemble_scalar(fem.form((u - u_avg_right) ** 2 * ds(markers.right))) / area_right), op=MPI.SUM)
    u_stdev_left = domain.comm.allreduce(np.sqrt(fem.assemble_scalar(fem.form((u - u_avg_left) ** 2 * ds(markers.left))) / area_left), op=MPI.SUM)
    eta_n = u_avg_left
    simulation_metadata = {
        "Negative Wagner Number": Wa_n,
        "Positive Wagner Number": Wa_p,
        "Negative Overpotential [V]": eta_n,
        "Positive Overpotential [V]": eta_p,
        "Open Circuit Potential (OCP) [V]": args.u_ocv,
        "Voltage": voltage,
        "interior penalty (gamma)": args.gamma,
        "interior penalty kr-modified (gamma)": gamma,
        "ionic to electronic conductivity ratio (kr)": args.kr,
        "average potential left [V]": u_avg_left,
        "stdev potential left [V]": u_stdev_left,
        "average potential right [V]": u_avg_right,
        "stdev potential right [V]": u_stdev_right,
        "Superficial current density [A/m2]": np.abs(i_sup),
        "Current at negative am - electrolyte boundary [A]": np.abs(I_neg_charge_xfer),
        "Current at electrolyte - positive am boundary [A]": np.abs(I_pos_am),
        "Current at right boundary [A]": np.abs(I_right),
        "Current at insulated boundary [A]": I_insulated,
        "stdev i positive charge transfer [A/m2]": std_dev_i_pos_am,
        "stdev i positive charge transfer (normalized)": std_dev_i_pos_am_norm,
        "solver atol": args.atol,
        "solver rtol": args.rtol,
        "Total resistance [Ω.cm2]": voltage / (np.abs(I_right) / (area_right * 1e4)),

    }
    # visualization
    if args.plot and comm.Get_size() == 1:
        n_points = 10000
        y_pos = [0.125, 0.5, 0.875]
        tol = 1e-14  # Avoid hitting the outside of the domain
        bb_trees = bb_tree(domain, domain.topology.dim)

        x = np.linspace(tol, LX - tol, n_points)
        points_top = np.zeros((3, n_points))
        points_mid = np.zeros((3, n_points))
        points_bot = np.zeros((3, n_points))

        points_top[0] = x
        points_top[1] = np.ones(n_points) * 0.875 * LY
        points_mid[0] = x
        points_mid[1] = np.ones(n_points) * 0.5 * LY
        points_bot[0] = x
        points_bot[1] = np.ones(n_points) * 0.125 * LY
        styles = ['r', 'b', 'g']
        # obtain values to plot
        u_values = []
        cells_top = []
        cells_mid = []
        cells_bot = []
        points_on_proc_top = []
        points_on_proc_mid = []
        points_on_proc_bot = []
        # Find cells whose bounding-box collide with the the points
        cell_candidates_top = compute_collisions_points(bb_trees, points_top.T)
        cell_candidates_mid = compute_collisions_points(bb_trees, points_mid.T)
        cell_candidates_bot = compute_collisions_points(bb_trees, points_bot.T)
        # Choose one of the cells that contains the point
        colliding_cells_top = compute_colliding_cells(domain, cell_candidates_top, points_top.T)
        colliding_cells_mid = compute_colliding_cells(domain, cell_candidates_mid, points_mid.T)
        colliding_cells_bot = compute_colliding_cells(domain, cell_candidates_bot, points_bot.T)

        for i in range(n_points):
            point_top = points_top.T[i]
            point_mid = points_mid.T[i]
            point_bot = points_bot.T[i]
            if len(colliding_cells_top.links(i)) > 0:
                points_on_proc_top.append(point_top)
                cells_top.append(colliding_cells_top.links(i)[0])

            if len(colliding_cells_mid.links(i)) > 0:
                points_on_proc_mid.append(point_mid)
                cells_mid.append(colliding_cells_mid.links(i)[0])

            if len(colliding_cells_bot.links(i)) > 0:
                points_on_proc_bot.append(point_bot)
                cells_bot.append(colliding_cells_bot.links(i)[0])

        points_on_proc_top = np.array(points_on_proc_top, dtype=np.float64)
        current_values_top = current_h.eval(points_on_proc_top, cells_top)

        points_on_proc_mid = np.array(points_on_proc_mid, dtype=np.float64)
        current_values_mid = current_h.eval(points_on_proc_mid, cells_mid)

        points_on_proc_bot = np.array(points_on_proc_bot, dtype=np.float64)
        current_values_bot = current_h.eval(points_on_proc_bot, cells_bot)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(points_on_proc_top[:, 0] / micron, np.linalg.norm(current_values_top, axis=1), styles[2], label='0.875' + r'$L_y$', linewidth=1)
        ax.plot(points_on_proc_mid[:, 0] / micron, np.linalg.norm(current_values_mid, axis=1), styles[1], label='0.5' + r'$L_y$', linewidth=1)
        ax.plot(points_on_proc_bot[:, 0] / micron, np.linalg.norm(current_values_bot, axis=1), styles[0], label='0.125' + r'$L_y$', linewidth=1)
        ax.grid(True)
        ax.legend()
        ax.set_xlim([0, LX / micron])
        # ax.set_ylim([0, voltage])
        ax.set_ylabel(r'$\Vert i \Vert$ [Am$^{-2}$]', rotation=90, labelpad=0, fontsize='xx-large')
        ax.set_xlabel(r'[$\mu$m]')
        ax.set_title(r'$\mathrm{Wa}$ = ' + f'{args.Wa_p}' + ',' + r'$\frac{\kappa}{\sigma}$ = ' + f'{args.kr}')
        plt.tight_layout()
        plt.savefig(current_dist_file)
        plt.show()
    if comm.rank == 0:
        utils.print_dict(simulation_metadata, padding=50)
        with open(simulation_metafile, "w", encoding='utf-8') as f:
            json.dump(simulation_metadata, f, ensure_ascii=False, indent=4)
        print(f"Saved results files in {workdir}")
        print(f"Time elapsed: {int(timeit.default_timer() - start_time):3.5f}s")
