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
import logging
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import ufl
import warnings

from basix.ufl import element, mixed_element
from dolfinx import cpp, default_real_type, default_scalar_type, fem, io, la, mesh, nls, plot
from dolfinx.fem import petsc
from dolfinx.io import gmshio, VTXWriter, XDMFFile
from dolfinx.nls import petsc as petsc_nls
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 dot, div, dx, ds, dS, grad, inner, grad, avg, jump)

import commons, configs, utils

warnings.simplefilter('ignore')


kappa_elec = 0.1
kappa_pos_am = 0.2
faraday_const = 96485
R = 8.3145
T = 298


def arctanh(y):
    return 0.5 * ufl.ln((1 + y) / (1 - y))


def ocv(c, cmax=35000):
    xi = 2 * (c - 0.5 * cmax) / cmax
    return 3.25 - 0.5 * arctanh(xi)

def initial_condition(x, val=0):
    values = np.zeros((1, x.shape[1]), dtype=default_scalar_type)
    values[0] = val
    return values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Current Collector.')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="lithium_metal_3d_cc_2d")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1.0, type=float)
    parser.add_argument("--Wa_n", help="Wagna number for negative electrode: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e-3, type=float)
    parser.add_argument("--Wa_p", help="Wagna number for positive electrode: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e3, type=float)
    parser.add_argument("--kr", help="ratio of ionic to electronic conductivity", nargs='?', const=1, default=1, type=float)
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
    workdir = os.path.join(args.mesh_folder, str(Wa_n) + "-" + str(Wa_p) + "-" + str(args.kr), str(args.gamma))
    utils.make_dir_if_missing(workdir)
    output_meshfile = os.path.join(args.mesh_folder, 'mesh.msh')
    lines_h5file = os.path.join(args.mesh_folder, 'lines.h5')
    potential_resultsfile = os.path.join(workdir, "potential.bp")
    concentration_resultsfile = os.path.join(workdir, "concentration.bp")
    current_dist_file = os.path.join(workdir, f"current-y-positions-{str(args.Wa_p)}-{str(args.kr)}.png")
    reaction_dist_file = os.path.join(workdir, f"reaction-dist-{str(args.Wa_p)}-{str(args.kr)}.png")
    current_resultsfile = os.path.join(workdir, "current.bp")
    simulation_metafile = os.path.join(workdir, "simulation.json")

    loglevel = configs.get_configs()['LOGGING']['level']
    dimensions = args.dimensions
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    formatter = logging.Formatter(f'%(levelname)s:%(asctime)s:{workdir}:%(message)s')
    fh = logging.FileHandler(os.path.basename(__file__).replace(".py", ".log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.debug(args.mesh_folder)

    markers = commons.Markers()

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
    P1 = element("DG", domain.basix_cell(), 1, dtype=default_real_type)
    P2 = element("DG", domain.basix_cell(), 1, dtype=default_real_type)
    # V = fem.functionspace(domain, mixed_element([P1, P2]))
    V = fem.functionspace(domain, ("DG", 1, (2,)))
    W = fem.functionspace(domain, ("DG", 1, (3,)))
    Z = fem.functionspace(domain, ("CG", 1, (3,)))
    Q = fem.functionspace(domain, ("DG", 0))
    uc = fem.Function(V, name='potential')

    u, c = ufl.split(uc)

    uc0 = fem.Function(V, name='potential')

    v, q = ufl.TestFunctions(V)

    current_h = fem.Function(W, name='current_density')
    kappa = fem.Function(Q, name='conductivity')
    D = fem.Function(Q, name='diffusivity')
    n = ufl.FacetNormal(domain)
    x = ufl.SpatialCoordinate(domain)
    h = ufl.CellDiameter(domain)
    h_avg = avg(h)

    cells_elec = ct.find(markers.electrolyte)
    kappa_pos_am = kappa_elec / args.kr
    cells_pos_am = ct.find(markers.positive_am)

    kappa.x.array[cells_elec] = np.full_like(cells_elec, kappa_elec, dtype=default_scalar_type)
    kappa.x.array[cells_pos_am] = np.full_like(cells_pos_am, kappa_pos_am, dtype=default_scalar_type)

    D.x.array[cells_pos_am] = np.full_like(cells_pos_am, 1e-15, dtype=default_scalar_type)
    D.x.array[cells_elec] = np.full_like(cells_elec, 1e-5, dtype=default_scalar_type)

    fun1 = lambda x: initial_condition(x, val=32500)
    fun2 = lambda x: initial_condition(x, val=0)
    uc0.sub(1).interpolate(fun1, cells=cells_pos_am)
    uc0.sub(1).interpolate(fun2, cells=cells_elec)
    uc0.x.scatter_forward()
    u0, c0 = ufl.split(uc0)
    uc.sub(1).interpolate(uc0.sub(1))

    dt = 1e-3
    TIME = 50 * dt
    voltage = args.voltage

    f = fem.Constant(domain, PETSc.ScalarType(0))
    g = fem.Constant(domain, PETSc.ScalarType(0))

    u_left = fem.Function(V).sub(0)
    with u_left.vector.localForm() as u0_loc:
        u0_loc.set(0)
    u_right = fem.Function(V).sub(0)
    with u_right.vector.localForm() as u1_loc:
        u1_loc.set(voltage)

    i0_n = kappa_elec * R * T / (args.Wa_n * faraday_const * LX)
    i0_p = kappa_elec * R * T / (args.Wa_p * faraday_const * LX)

    u_ocv = ocv(c("+"))
    V_left = 0

    alpha = 100#args.gamma
    gamma = 100#args.gamma
    i_loc = -inner((kappa * grad(u))('+'), n("+"))
    u_jump = 2 * ufl.ln(0.5 * i_loc/i0_p + ufl.sqrt((0.5 * i_loc/i0_p)**2 + 1)) * (R * T / faraday_const)

    Fu = kappa * inner(grad(u), grad(v)) * dx - f * v * dx - kappa * inner(grad(u), n) * v * ds

    # Add DG/IP terms
    Fu += - avg(kappa) * inner(jump(u, n), avg(grad(v))) * dS#(0)
    # F += - inner(utils.jump(kappa * u, n), avg(grad(v))) * dS(0)
    Fu += - inner(avg(kappa * grad(u)), jump(v, n)) * dS#(0)
    # F += + avg(u) * inner(utils.jump(kappa, n), avg(grad(v))) * dS(0)
    Fu += alpha / h_avg * avg(kappa) * inner(jump(v, n), jump(u, n)) * dS#(0)

    # Internal boundary
    Fu += + avg(kappa) * dot(avg(grad(v)), (u_jump + u_ocv) * n('+')) * dS(markers.electrolyte_v_positive_am)
    Fu += -alpha / h_avg * avg(kappa) * dot(jump(v, n), (u_jump + u_ocv) * n('+')) * dS(markers.electrolyte_v_positive_am)

    # # Symmetry
    Fu += - avg(kappa) * inner(jump(u, n), avg(grad(v))) * dS(markers.electrolyte_v_positive_am)

    # # Coercivity
    Fu += alpha / h_avg * avg(kappa) * inner(jump(u, n), jump(v, n)) * dS(markers.electrolyte_v_positive_am)

    # Nitsche Dirichlet BC terms on left and right boundaries
    Fu += - kappa * (u - u_left) * inner(n, grad(v)) * ds(markers.left)
    Fu += -gamma / h * (u - u_left) * v * ds(markers.left)
    Fu += - kappa * (u - u_right) * inner(n, grad(v)) * ds(markers.right) 
    Fu += -gamma / h * (u - u_right) * v * ds(markers.right)

    # Nitsche Neumann BC terms on insulated boundary
    Fu += -g * v * ds(markers.insulated_electrolyte) + gamma * h * g * inner(grad(v), n) * ds(markers.insulated_electrolyte)
    Fu += - gamma * h * inner(inner(grad(u), n), inner(grad(v), n)) * ds(markers.insulated_electrolyte)
    Fu += -g * v * ds(markers.insulated_positive_am) + gamma * h * g * inner(grad(v), n) * ds(markers.insulated_positive_am)
    Fu += - gamma * h * inner(inner(grad(u), n), inner(grad(v), n)) * ds(markers.insulated_positive_am)

    # kinetics boundary - neumann
    # Fu += - gamma * h * inner(inner(kappa * grad(u), n), inner(grad(v), n)) * ds(markers.left)
    # Fu -= - gamma * h * 2 * i0_n * ufl.sinh(0.5 * faraday_const / R / T * (V_left - u - 0)) * inner(grad(v), n) * ds(markers.left)

    Fc = c * q * dx - c0 * q * dx

    Fct = D * inner(grad(c), grad(q)) * dx - f * q * dx - D * inner(grad(c), n) * q * ds

    # Add DG/IP terms
    Fct += - avg(D) * inner(jump(c, n), avg(grad(q))) * dS#(0)
    Fct += - inner(avg(D * grad(c)), jump(q, n)) * dS#(0)
    Fct += alpha / h_avg * avg(D) * inner(jump(q, n), jump(c, n)) * dS#(0)

    # zero-concentration
    Fct += - D * (c - 0) * inner(n, grad(q)) * ds(markers.left)
    Fct += -gamma / h * (c - 0) * q * ds(markers.left)

    # insulated
    Fct += -g * q * ds(markers.insulated_electrolyte) + gamma * h * g * inner(grad(q), n) * ds(markers.insulated_electrolyte)
    Fct += - gamma * h * inner(inner(grad(c), n), inner(grad(q), n)) * ds(markers.insulated_electrolyte)
    Fct += -g * q * ds(markers.insulated_positive_am) + gamma * h * g * inner(grad(q), n) * ds(markers.insulated_positive_am)
    Fct += - gamma * h * inner(inner(grad(c), n), inner(grad(q), n)) * ds(markers.insulated_positive_am)
    Fct += -g * q * ds(markers.right) + gamma * h * g * inner(grad(q), n) * ds(markers.right)
    Fct += - gamma * h * inner(inner(grad(c), n), inner(grad(q), n)) * ds(markers.right)

    # Internal boundary
    Fct += - (1/faraday_const) * inner(jump(q, n), avg(kappa * grad(u))) * dS(markers.electrolyte_v_positive_am)
    # Fct += -alpha / h_avg * avg(D) * dot(jump(q, n), (u_jump + u_ocv) * n('+')) * dS(markers.electrolyte_v_positive_am)

    # # # Symmetry
    Fct += - avg(D) * inner(jump(c, n), avg(grad(q))) * dS(markers.electrolyte_v_positive_am)

    # # # Coercivity
    Fct += alpha / h_avg * avg(D) * inner(jump(c, n), jump(q, n)) * dS(markers.electrolyte_v_positive_am)

    Fc += dt * Fct

    F = Fu + Fc

    problem = petsc.NonlinearProblem(F, uc)
    solver = petsc_nls.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    # solver.maximum_iterations = 100
    # solver.rtol = args.rtol
    # solver.atol = args.atol

    ksp = solver.krylov_solver

    opts = PETSc.Options()
    ksp.setMonitor(lambda _, it, residual: print(it, residual))
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"

    ksp.setFromOptions()
    t = 0.0
    c_vtx = VTXWriter(comm, concentration_resultsfile, uc, engine="BP5")
    c_vtx.write(0.0)

    while t < TIME:
        t += dt
        logger.debug(f"Time: {t}")
        n_iters, converged = solver.solve(uc)
        uc.x.scatter_forward()
        uc0.x.array[:] = uc.x.array
        I_pos_am_pot = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(-(kappa * grad(u))("+"), n("+")) * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)
        I_pos_am_conc = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(-(faraday_const * D * grad(c))("+"), n("+")) * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)
        logger.debug(f"Charge transfer current (potential): {I_pos_am_pot:.4f} A, Charge transfer current (concentration): {I_pos_am_conc:.4f} A")
        c_vtx.write(t)
    c_vtx.close()
