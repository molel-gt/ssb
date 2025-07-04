# SPDX-License-Identifier: MIT
import argparse
import datetime
import json
import logging
import os
import resource
import time
import timeit

import basix
import dolfinx
import dolfinx.fem.petsc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.special as sp
import ufl
import warnings

from dolfinx import cpp, default_real_type, fem, io, jit, mesh, log
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.nls import petsc as petsc_nls
from matplotlib import rc
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
from ufl import dot, grad, inner

import commons, constants, mesh_utils, plot_opts, solvers, solver_params, utils

warnings.simplefilter("ignore")
plt.rcParams.update(plot_opts.params)
# logging.basicConfig(level=logging.INFO)

R = 8.314
T = 298
faraday_const = 96485
kappa_pos_am = 0.1
kinetics = ('linear', 'tafel', 'butler_volmer')
micron = 1e-6
V_UCO = 5.0  # upper cutoff voltage
c_max = 35000
directions = {'x': 0, 'y': 1, 'z': 2}


log.set_log_level(log.LogLevel.WARNING)


def compute_cell_boundary_facets(domain, ct, marker):
    """Compute the integration entities for integrals around the
    boundaries of all cells in domain.

    Parameters:
        domain: The mesh.
        ct: cell tags
        marker: physical group label

    Returns:
        Facets to integrate over, identified by ``(cell, local facet
        index)`` pairs.
    """
    tdim = domain.topology.dim
    fdim = tdim - 1
    n_f = cpp.mesh.cell_num_entities(domain.topology.cell_type, fdim)

    cells_1 = ct.find(marker)
    perm = np.argsort(cells_1)
    n_c = cells_1.shape[0]

    return np.vstack((np.repeat(cells_1[perm], n_f), np.tile(np.arange(n_f), n_c))).T.flatten()


class SolverTypes:
    def __init__(self):
        pass

    @property
    def direct(self):
        return "direct"

    @property
    def nested_iterative(self):
        return "nested_iterative"

    @property
    def block_iterative(self):
        return "block_iterative"

    @property
    def block_gs(self):
        return "block_gs"


def define_interior_eq(domain, degree,  submesh, submesh_to_mesh, value, kappa, cell_type):
    # Compute map from parent entity to submesh cell
    codim = domain.topology.dim - submesh.topology.dim
    ptdim = domain.topology.dim - codim
    num_entities = (
        domain.topology.index_map(ptdim).size_local
        + domain.topology.index_map(ptdim).num_ghosts
    )
    mesh_to_submesh = np.full(num_entities, -1)
    mesh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh), dtype=np.int32)

    # el = basix.ufl.element(basix.ElementFamily.P, cell_type, degree, basix.LagrangeVariant.gll_warped, dtype=dolfinx.default_real_type)
    el = ("CG", degree)
    V = fem.functionspace(submesh, el)
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    ct_r = mesh.meshtags(domain, domain.topology.dim, submesh_to_mesh, np.full_like(submesh_to_mesh, 1, dtype=np.int32))
    val = fem.Constant(submesh, value)
    dx_r = ufl.Measure("dx", domain=domain, subdomain_data=ct_r, subdomain_id=1)
    F = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_r #- val * v * dx_r
    return u, F, mesh_to_submesh


def mixed_term(u, v, n):
    return ufl.dot(ufl.grad(u), n) * v


def surface_overpotential(kappa, u, n, i0, kinetics_type='linear', ref={"L": 1, "phi": 1, "t": 1, "c": 1}):
    i_loc = -inner((kappa * grad(u)), n) * ref["phi"]/ref["L"]
    if kinetics_type == "butler_volmer":
        return 2 * ufl.ln(0.5 * i_loc/i0 + ufl.sqrt((0.5 * i_loc/i0)**2 + 1)) * (R * T / (faraday_const * ref["phi"]))
    elif kinetics_type == "linear":
        return R * T * i_loc / (i0 * faraday_const * ref["phi"])
    elif kinetics_type == "tafel":
        return ufl.sign(i_loc) * R * T / (0.5 * faraday_const * ref["phi"]) * ufl.ln(np.abs(i_loc)/i0)


def arctanh(y):
    return 0.5 * ufl.ln((1 + y) / (1 - y))


def ocv_chen2020(c, cmax):
    return  4.4875 - 0.8090 * c/cmax - 0.0428 * ufl.tanh(18.5138*(c/cmax - 0.5542)) +\
    -17.7326 * ufl.tanh(15.7890*(c/cmax - 0.3117)) + 17.5842 * ufl.tanh(15.9308*(c/cmax - 0.3120))


def get_Lref(dimensions, transport_direction):
    direction = directions[transport_direction.lower()]

    return dimensions[direction]


def cross_section_area(dims, transport_direction):
    direction = directions[transport_direction.lower()]
    values = [0, 1, 2]
    values.pop(direction)
    if np.isclose(dims[values[0]], 0):
        return dims[values[1]]
    elif np.isclose(dims[values[1]], 0):
        return dims[values[0]]
    elif np.isclose(dims[values[0]], 0) and np.isclose(dims[values[1]], 0):
        raise ValueError("Invalid")
    return dims[values[0]] * dims[values[1]]


def get_eigenvalues(M):
    Print = PETSc.Sys.Print
    for eps_type in [SLEPc.EPS.Which.SMALLEST_MAGNITUDE, SLEPc.EPS.Which.LARGEST_MAGNITUDE]:
        E = SLEPc.EPS()
        E.create(M.getComm().tompi4py())
        E.setOperators(M)
        E.setWhichEigenpairs(eps_type)
        opts = PETSc.Options()
        E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        E.setFromOptions()
        E.solve()
        nconv = E.getConverged()

        vw = PETSc.Viewer.STDOUT()
        if nconv>0:
            sx, _ = M.createVecs()
            E.getEigenpair(0, sx)
            vw.pushFormat(PETSc.Viewer.Format.ASCII_INFO_DETAIL)
            E.errorView(viewer=vw)
        else:
            Print( "No eigenpairs converged" )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='secondary current distribution')
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1.0, type=float)
    parser.add_argument("--u_ocv", help="open-circuit potential", nargs='?', const=1, default=0, type=float)
    parser.add_argument("--Wa_n", help="Wagna number for negative electrode: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e-3, type=float)
    parser.add_argument("--Wa_p", help="Wagna number for positive electrode: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e3, type=float)
    parser.add_argument("--kr", help="ratio of ionic to electronic conductivity", nargs='?', const=1, default=1, type=float)
    parser.add_argument("--gamma", help="interior penalty parameter", nargs='?', const=1, default=15, type=float)
    parser.add_argument("-p", "--p", help="polynomial approximation order", nargs='?', const=1, default=4, type=int)
    parser.add_argument("-cell_type", "--cell_type", help="cell type to use", nargs='?', const=1, default="tetrahedron", type=str)
    parser.add_argument("-dt", "--dt", help="minimum normalized time step", nargs='?', const=1, default=2e-7, type=float)
    parser.add_argument("--atol", help="solver absolute tolerance", nargs='?', const=1, default=1e-12, type=float)
    parser.add_argument("--rtol", help="solver relative tolerance", nargs='?', const=1, default=1e-9, type=float)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='MICRON_TO_METER', type=str)
    parser.add_argument('--solver_type', help='solver type to use', nargs='?',
                        const=1, default='direct', type=str)
    parser.add_argument('--amg_type', help='if using iterative solver, which algebraic multigrid type to use', nargs='?',
                        const=1, default='gamg', type=str)
    parser.add_argument('--transport_direction', help='direction perpendicular to current collectors', nargs='?', const=1, default='X', type=str)
    parser.add_argument('--kinetics', help='kinetics type', nargs='?', const=1, default='butler_volmer', type=str, choices=kinetics)
    parser.add_argument("--plot", help="whether to plot results", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot_sparsity", help="whether to plot results", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--improved_guess", help="whether to solve for improved guess", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    start_time = timeit.default_timer()
    voltage = args.voltage
    Wa_n = args.Wa_n
    Wa_p = args.Wa_p
    gamma = args.gamma
    kappa_elec = args.kr * kappa_pos_am
    dt_ = args.dt
    D = 1e-15
    TIME = 1 * dt_

    markers = commons.Markers()
    solver_types = SolverTypes()
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    cell_type = getattr(basix.CellType, args.cell_type)

    dimensions = utils.extract_dimensions_from_meshfolder(args.mesh_folder)
    LX, LY, LZ = [float(vv) * micron for vv in dimensions.split("-")]

    L_ref = get_Lref([LX, LY, LZ], args.transport_direction)
    A0 = cross_section_area([LX, LY, LZ], args.transport_direction) * 1e4  # [cm^2]

    # reference values
    t_ref = L_ref ** 2 / D
    phi_ref = V_UCO
    # c_ref = c_max
    c_ref = kappa_elec * phi_ref / (faraday_const * D)
    ref = {"t": t_ref, "phi": phi_ref, "c": c_ref, "L": L_ref}

    soc_init = 0.75 * c_max / c_ref

    output_meshfile = os.path.join(args.mesh_folder, "mesh.msh")
    results_dir = os.path.join(args.mesh_folder, args.kinetics, str(Wa_n) + "-" + str(Wa_p) + "-" + str(args.kr), str(args.gamma), str(comm.Get_size()))
    utils.make_dir_if_missing(results_dir)
    output_potential_file = os.path.join(results_dir, "potential.bp")
    elec_potential_file = os.path.join(results_dir, "electrolyte_potential.bp")
    positive_am_potential_file = os.path.join(results_dir, "positive_am_potential.bp")
    current_file = os.path.join(results_dir, "current.bp")
    concentration_file = os.path.join(results_dir, "concentration.bp")
    potential_plot_file = os.path.join(results_dir, "potential.eps")
    concentration_plot_file = os.path.join(results_dir, "concentration.eps")
    simulation_metafile = os.path.join(results_dir, "simulation.json")
    convergence_history = os.path.join(results_dir, "convergence.eps")
    log_datafile = os.path.join(results_dir, "log.txt")

    # load mesh
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = io.gmshio.read_from_msh(output_meshfile, comm, partitioner=partitioner)[:3]
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)
    domain.topology.create_connectivity(tdim, tdim)
    domain.topology.create_connectivity(fdim, fdim)

    # tag internal facets as 0
    ft_imap = domain.topology.index_map(fdim)
    num_facets_local = ft_imap.size_local + ft_imap.num_ghosts

    # facets
    num_facets_local = (
        domain.topology.index_map(fdim).size_local + domain.topology.index_map(fdim).num_ghosts
    )
    facets = np.arange(num_facets_local, dtype=np.int32)
    values = np.full_like(facets, 0, dtype=np.int32)
    values[ft.find(markers.left)] = markers.left
    values[ft.find(markers.right)] = markers.right
    all_b_facets = mesh.compute_incident_entities(
        domain.topology, ct.find(markers.electrolyte), tdim, fdim
    )
    all_t_facets = mesh.compute_incident_entities(
        domain.topology, ct.find(markers.positive_am), tdim, fdim
    )
    interface = np.intersect1d(all_b_facets, all_t_facets)
    values[interface] = markers.electrolyte_v_positive_am

    ft = mesh.meshtags(domain, fdim, facets, values)

    submesh_electrolyte, submesh_electrolyte_to_mesh, b_v_map = mesh.create_submesh(
        domain, tdim, ct.find(markers.electrolyte))[0:3]
    submesh_positive_am, submesh_positive_am_to_mesh, t_v_map = mesh.create_submesh(
        domain, tdim, ct.find(markers.positive_am))[0:3]
    submesh_pos_am_facets, submesh_pos_am_facets_to_mesh, f_v_map = mesh.create_submesh(
        domain, fdim, facets)[:3]
    # submesh_pos_am_facets, submesh_pos_am_facets_to_mesh, f_v_map = mesh.create_submesh(
    #     domain, fdim, ct.find(markers.positive_am))[:3]
    submesh_pos_am_facets.topology.create_connectivity(fdim, fdim)
    parent_to_sub_electrolyte = np.full(num_facets_local, -1, dtype=np.int32)
    parent_to_sub_electrolyte[submesh_electrolyte_to_mesh] = np.arange(len(submesh_electrolyte_to_mesh), dtype=np.int32)
    parent_to_sub_positive_am = np.full(num_facets_local, -1, dtype=np.int32)
    parent_to_sub_positive_am[submesh_positive_am_to_mesh] = np.arange(len(submesh_positive_am_to_mesh), dtype=np.int32)
    submesh_positive_am.topology.create_entities(fdim)
    num_facets_local = (
        domain.topology.index_map(fdim).size_local + domain.topology.index_map(fdim).num_ghosts
    )
    parent_to_pos_am_facets = np.full(num_facets_local, -1, dtype=np.int32)
    parent_to_pos_am_facets[submesh_pos_am_facets_to_mesh] = np.arange(len(submesh_pos_am_facets_to_mesh), dtype=np.int32)

    ft_electrolyte = mesh_utils.transfer_meshtags(domain, submesh_electrolyte, submesh_electrolyte_to_mesh, ft)
    ft_positive_am = mesh_utils.transfer_meshtags(domain, submesh_positive_am, submesh_positive_am_to_mesh, ft)
    # ft_pos_am_facets = mesh_utils.transfer_meshtags(domain, submesh_pos_am_facets, submesh_pos_am_facets_to_mesh, ft)


    # Hack, as we use one-sided restrictions, pad dS integral with the same entity from the same cell on both sides
    domain.topology.create_connectivity(fdim, tdim)
    # domain.topology.create_connectivity(fdim, fdim-1)
    submesh_pos_am_facets.topology.create_connectivity(fdim, fdim-1)
    # submesh_pos_am_facets.topology.create_connectivity(tdim, fdim)
    f_to_c = domain.topology.connectivity(fdim, tdim)

    for facet in ft.find(markers.electrolyte_v_positive_am):
        cells = f_to_c.links(facet)
        assert len(cells) == 2
        b_map = parent_to_sub_electrolyte[cells]
        t_map = parent_to_sub_positive_am[cells]
        f_map = parent_to_pos_am_facets[cells]
        parent_to_sub_electrolyte[cells] = max(b_map)
        parent_to_sub_positive_am[cells] = max(t_map)
        parent_to_pos_am_facets[cells] = max(f_map)

    entity_maps = {
                    submesh_electrolyte: parent_to_sub_electrolyte,
                    submesh_positive_am: parent_to_sub_positive_am,
                    submesh_pos_am_facets: parent_to_pos_am_facets
                    }

    u_0, F_00, m_to_elec = define_interior_eq(domain, 1, submesh_electrolyte, submesh_electrolyte_to_mesh, 0.0, kappa_elec, cell_type)
    u_1, F_11, m_to_pos_am = define_interior_eq(domain, 1, submesh_positive_am, submesh_positive_am_to_mesh, 0.0, kappa_pos_am, cell_type)
    u_0.name = "u_b"
    u_1.name = "u_t"

    # initial guess
    u_0.interpolate(lambda x: x[directions[args.transport_direction.lower()]]*0.95)
    u_1.interpolate(lambda x: 1.05*x[directions[args.transport_direction.lower()]]/1.1)

    # Add coupling term to the interface
    # Get interface markers on submesh b
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
    int_facet_domains = [(markers.electrolyte_v_positive_am, int_facet_domain)]

    dInterface = ufl.Measure("dS", domain=domain, subdomain_data=int_facet_domains, subdomain_id=markers.electrolyte_v_positive_am)
    dx_r = ufl.Measure('dx', domain=domain, subdomain_data=ct, subdomain_id=markers.positive_am)
    dx_c = ufl.Measure('dx', domain=submesh_positive_am)
    ds = ufl.Measure('ds', domain=domain, subdomain_data=ft)
    ds_c = ufl.Measure('ds', domain=submesh_positive_am, subdomain_data=ft_positive_am)
    l_res = "-"
    r_res = "+"
    V0 = u_0.function_space
    V1 = u_1.function_space

    v_l = ufl.TestFunction(V0)(l_res)
    v_r = ufl.TestFunction(V1)(r_res)
    u_l = u_0(l_res)
    u_r = u_1(r_res)

    n = ufl.FacetNormal(domain)
    n_c = ufl.FacetNormal(submesh_positive_am)
    n_l = n(l_res)
    n_r = n(r_res)
    cd = ufl.CellDiameter(domain)
    h_l = cd(l_res)
    h_r = cd(r_res)

    # exchange current densities
    i0_n = kappa_elec * R * T / (Wa_n * faraday_const * L_ref)
    i0_p = kappa_elec * R * T / (Wa_p * faraday_const * L_ref)

    # concentration problem
    dt = fem.Constant(submesh_positive_am, dt_)
    # el = basix.ufl.element(basix.ElementFamily.P, cell_type, args.p, basix.LagrangeVariant.gll_isaac, dtype=dolfinx.default_real_type)
    el = ("DG", args.p)
    VC = fem.functionspace(submesh_positive_am, el)
    VCbar = fem.functionspace(submesh_pos_am_facets, el)

    c, q = fem.Function(VC), ufl.TestFunction(VC)
    cbar, qbar = fem.Function(VCbar), ufl.TestFunction(VCbar)
    c0 = fem.Function(VC)
    u_int = fem.Function(VC)

    c0.interpolate(lambda x: x[directions[args.transport_direction.lower()]] - x[directions[args.transport_direction.lower()]] + soc_init)
    # c.interpolate(c0)#lambda x: soc_init * (1 - np.exp(-x[directions[args.transport_direction.lower()]])))

    ##################### hdg elements for concentration ##############################
    ### cell and facet mesh
    cell_boundary_facets = compute_cell_boundary_facets(domain, ct, markers.positive_am)
    cell_boundaries = 99  # A tag
    # Create the measure
    ds_fc = ufl.Measure("ds", subdomain_data=[(cell_boundaries, cell_boundary_facets)], domain=domain)
    # Create a cell integral measure over the facet mesh
    dx_f = ufl.Measure("dx", domain=submesh_pos_am_facets)

    # add to entity maps
    gamma_r = 16.0e-3 * args.p**2 / ufl.CellDiameter(submesh_positive_am)  # Scaled penalty parameter

    q_r = ufl.TestFunction(c.function_space)(r_res)
    q_l = ufl.TestFunction(c.function_space)(l_res)
    c_r = c(r_res)

    jump_u = surface_overpotential(kappa_pos_am, u_r, n_r, i0_p, kinetics_type=args.kinetics, ref=ref) + ocv_chen2020(c(r_res), cmax=c_max/c_ref)/phi_ref

    F_0 = (
        - 0.5 * mixed_term(kappa_elec * u_l + kappa_pos_am * u_r, v_l, n_l) * dInterface
        - 0.5 * mixed_term(0.5 * (kappa_elec + kappa_pos_am) * v_l, (u_r - u_l - jump_u), n_l) * dInterface
    )

    F_1 = (
        + 0.5 * mixed_term(kappa_elec * u_l + kappa_pos_am * u_r, v_r, n_l) * dInterface
        - 0.5 * mixed_term(0.5 * (kappa_elec + kappa_pos_am) * v_r, (u_r - u_l - jump_u), n_l) * dInterface
    )
    F_0 += -2 * gamma / (h_l + h_r) * 0.5 * (kappa_elec + kappa_pos_am) * (u_r - u_l - jump_u) * v_l * dInterface
    F_1 += +2 * gamma / (h_l + h_r) * 0.5 * (kappa_elec + kappa_pos_am) * (u_r - u_l - jump_u) * v_r * dInterface

    F_0 += F_00
    F_1 += F_11

    # F_2 = (c - c0)/dt * q * dx_r + inner(ufl.grad(c), ufl.grad(q)) * dx_r
    # # F_2 += -inner(kappa_pos_am * phi_ref/(D * faraday_const * c_ref) * grad(u_r), n_r) * q_r * dInterface
    # F_2 += -inner(grad(u_r), n_r) * q_r * dInterface

    F_2a = (c - c0)/dt * q * dx_c + inner(grad(c), grad(q)) * dx_c
    F_2a += - inner(c - cbar, inner(grad(q), n)) * ds_fc(99)
    F_2a += + inner(grad(c), n) * q * ds_fc(99)
    F_2a += inner(grad(c), n) * q * ds_fc(99)
    F_2a = + gamma_r * inner(c - cbar, q) * ds_fc(99)
    F_2a += -inner(grad(u_r), n_r) * q_r * dInterface

    F_2b = inner(grad(c), n) * qbar * ds_fc(99)
    F_2b += - gamma_r * inner(c - cbar, qbar) * ds_fc(99)
    # F_2b += -inner(grad(u_r), n_r) * qbar(r_res) * dInterface

    jac00 = ufl.derivative(F_0, u_0)
    jac01 = ufl.derivative(F_0, u_1)
    jac02 = ufl.derivative(F_0, c)
    jac03 = ufl.derivative(F_0, cbar)

    jac10 = ufl.derivative(F_1, u_0)
    jac11 = ufl.derivative(F_1, u_1)
    jac12 = ufl.derivative(F_1, c)
    jac13 = ufl.derivative(F_1, cbar)

    jac20 = ufl.derivative(F_2a, u_0)
    jac21 = ufl.derivative(F_2a, u_1)
    jac22 = ufl.derivative(F_2a, c)
    jac23 = ufl.derivative(F_2a, cbar)

    jac30 = ufl.derivative(F_2b, u_0)
    jac31 = ufl.derivative(F_2b, u_1)
    jac32 = ufl.derivative(F_2b, c)
    jac33 = ufl.derivative(F_2b, cbar)

    J00 = fem.form(jac00, entity_maps=entity_maps)
    J01 = fem.form(jac01, entity_maps=entity_maps)
    J02 = fem.form(jac02, entity_maps=entity_maps)
    J03 = fem.form(jac03, entity_maps=entity_maps)

    J10 = fem.form(jac10, entity_maps=entity_maps)
    J11 = fem.form(jac11, entity_maps=entity_maps)
    J12 = fem.form(jac12, entity_maps=entity_maps)
    J13 = fem.form(jac13, entity_maps=entity_maps)

    J20 = fem.form(jac20, entity_maps=entity_maps)
    J21 = fem.form(jac21, entity_maps=entity_maps)
    J22 = fem.form(jac22, entity_maps=entity_maps)
    J23 = fem.form(jac23, entity_maps=entity_maps)

    J30 = fem.form(jac30, entity_maps=entity_maps)
    J31 = fem.form(jac31, entity_maps=entity_maps)
    J32 = fem.form(jac32, entity_maps=entity_maps)
    J33 = fem.form(jac33, entity_maps=entity_maps)
    
    J = [[J00, J01, J02, J03], [J10, J11, J12, J13], [J20, J21, J22, J23], [J30, J31, J32, J33]]
    P = J

    V0_map = V0.dofmap.index_map
    V1_map = V1.dofmap.index_map
    VC_map = VC.dofmap.index_map
    VCbar_map = VCbar.dofmap.index_map
    V0_dofmap = V0.dofmap
    V1_dofmap = V1.dofmap
    VC_dofmap = VC.dofmap
    VCbar_dofmap = VCbar.dofmap

    F = [
        fem.form(F_0, entity_maps=entity_maps),
        fem.form(F_1, entity_maps=entity_maps),
        fem.form(F_2a, entity_maps=entity_maps),
        fem.form(F_2b, entity_maps=entity_maps),
    ]
    left_bc = fem.Function(V0)
    left_bc.x.array[:] = 0/phi_ref
    submesh_electrolyte.topology.create_connectivity(
        submesh_electrolyte.topology.dim - 1, submesh_electrolyte.topology.dim
    )
    bc_left = fem.dirichletbc(
        left_bc, fem.locate_dofs_topological(u_0.function_space, fdim, ft_electrolyte.find(markers.left))
    )

    right_bc = fem.Function(V1)
    right_bc.x.array[:] = args.voltage/phi_ref
    submesh_positive_am.topology.create_connectivity(
        submesh_positive_am.topology.dim - 1, submesh_positive_am.topology.dim
    )
    bc_right = fem.dirichletbc(
        right_bc, fem.locate_dofs_topological(u_1.function_space, fdim, ft_positive_am.find(markers.right))
    )
    bcs = [bc_left, bc_right]

    local_dofs_u0 = np.setdiff1d(V0_map.local_to_global(np.arange(V0_map.size_local + V0_map.num_ghosts,
                                                                  dtype=np.int32)),
                                 V0_map.ghosts)
    local_dofs_u1 = np.setdiff1d(V1_map.local_to_global(np.arange(V1_map.size_local + V1_map.num_ghosts, dtype=np.int32)),
                                 V1_map.ghosts)
    local_dofs_u = np.sort(np.hstack((local_dofs_u0, local_dofs_u1)))
    local_dofs_c = np.sort(np.setdiff1d(VC_map.local_to_global(np.arange(VC_map.size_local + VC_map.num_ghosts, dtype=np.int32)), VC_map.ghosts))
    offset_u1 = V0_map.size_local*V0.dofmap.index_map_bs
    offset_c = V0_map.size_local*V0.dofmap.index_map_bs + V1_map.size_local*V1.dofmap.index_map_bs
    n_dofs = V0_map.size_global*V0.dofmap.index_map_bs + V1_map.size_global*V1.dofmap.index_map_bs + VC_map.size_global*VC.dofmap.index_map_bs +\
     VCbar_map.size_global*VCbar.dofmap.index_map_bs

    t = 0
    cvtx = io.VTXWriter(comm, concentration_file, [c], engine="BP5")
    PETSc.Sys.Print(f"Setting up problem Wa: {args.Wa_p}, Kr: {args.kr}, #DoFs: {n_dofs:,}, nprocs: {comm.Get_size()}")

    log_viewer = PETSc.Viewer().STDOUT()
    log_viewer.setFileName(log_datafile)

    while t < TIME:
        t += dt.value
        PETSc.Sys.Print(f"Time: {t:.1e}\n")
        if args.solver_type == solver_types.direct:
            opts = {
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                'pc_factor_mat_solver_type': 'mumps',
                }
            opts['log_view'] = None
            solver = solvers.NewtonSolver(
                F,
                J,
                [u_0, u_1, c, cbar],
                bcs=bcs,
                max_iterations=1000,
                petsc_options=opts,
                )
            PETSc.Log().begin()
            t0 = time.time()
            solver.solve()
            t1 = time.time()
            PETSc.Log().view(log_viewer)

        elif args.solver_type == solver_types.block_iterative:
            Jmat = fem.petsc.create_matrix_nest(J)
            nested_IS = Jmat.getNestISs()
            IS_u0 = nested_IS[0][0]
            IS_u1 = nested_IS[0][1]
            IS_u = IS_u0.sum(IS_u1)
            IS_c = nested_IS[0][2].sum(nested_IS[0][3])
            Jmat = fem.petsc.create_matrix_block(J)
            Pmat = fem.petsc.create_matrix_block(P)
            Fvec = fem.petsc.create_vector_block(F)
            snes = PETSc.SNES().create(comm)
            snes.setType('newtonls')
            snes.setTolerances(rtol=1.0e-7, max_it=15)
            snes.getKSP().setType(PETSc.KSP.Type.PREONLY)
            snes.getKSP().setOptionsPrefix("snes_")
            snes.getKSP().setOperators(Jmat, Pmat)
            nullspace = PETSc.NullSpace().create(constant=True)
            PETSc.Mat.setNearNullSpace(Jmat, nullspace)
            snes.getKSP().setTolerances(rtol=1e-7, max_it=100)
            snes.setErrorIfNotConverged(True)
            snes.getKSP().setErrorIfNotConverged(True)
            snes.getKSP().setConvergenceHistory()
            snes.getKSP().getPC().setType("fieldsplit")
            snes.getKSP().getPC().setFieldSplitIS(("u", IS_u), ("c", IS_c))
            opts = PETSc.Options()
            for kopt, vopt in solver_params.LINESEARCH.items():
                opts[kopt] = vopt

            opts['log_view'] = None

            opts[f"{snes.getKSP().getOptionsPrefix()}pc_fieldsplit_off_diag_use_amat"] = True
            opts[f"{snes.getKSP().getOptionsPrefix()}pc_fieldsplit_detect_saddle_point"] = True

            ksp_u, ksp_c = snes.getKSP().getPC().getFieldSplitSubKSP()

            snes.getKSP().getPC().setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
            snes.getKSP().getPC().setFieldSplitSchurPreType(PETSc.PC.SchurPreType.SELFP)
            snes.getKSP().getPC().setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)

            ksp_u.setType(PETSc.KSP.Type.PREONLY)
            ksp_u.getPC().setType(PETSc.PC.Type.ILU)
            ksp_u.setTolerances(rtol=1e-7, max_it=100)
            opts[f"{ksp_u.getOptionsPrefix()}pc_factor_levels"] = 0
            opts[f"{ksp_u.getOptionsPrefix()}pc_factor_fill"] = 2.0

            ksp_c.setType(PETSc.KSP.Type.CG)
            ksp_c.getPC().setType("lu")#args.amg_type)
            ksp_c.getPC().setFactorSolverType("superlu_dist")
            # ksp_c.getPC().setFieldSplitIS(("0", nested_IS[0][2]), ("1", nested_IS[0][3]))
            # ksp_c0, ksp_c1 = ksp_c.getPC().getFieldSplitSubKSP()
            # ksp_c.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
            # # ksp_c.getPC().setFieldSplitSchurPreType(PETSc.PC.SchurPreType.SELFP)
            # # ksp_c.getPC().setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)

            # ksp_c0.setType(PETSc.KSP.Type.FGMRES)
            # ksp_c0.getPC().setType(PETSc.PC.Type.ILU)
            # ksp_c1.setType(PETSc.KSP.Type.PREONLY)
            # ksp_c1.getPC().setType(PETSc.PC.Type.LU)

            # opts[f"{ksp_c0.getOptionsPrefix()}pc_factor_levels"] = 0
            # opts[f"{ksp_c0.getOptionsPrefix()}pc_factor_fill"] = 2.0
            # opts[f"{ksp_c1.getOptionsPrefix()}pc_factor_levels"] = 0
            # opts[f"{ksp_c1.getOptionsPrefix()}pc_factor_fill"] = 2.0
            ksp_c.setTolerances(rtol=1e-7, max_it=100)
            ksp_u.setConvergenceHistory()
            ksp_c.setConvergenceHistory()

            # opts[f'{ksp_u.getOptionsPrefix()}ksp_monitor_singular_value'] = None
            opts[f'{ksp_c.getOptionsPrefix()}ksp_monitor_singular_value'] = None
            opts[f"{ksp_c.getOptionsPrefix()}mat_schur_complement_ainv_type"] = "lump"
            opts[f"{ksp_c.getOptionsPrefix()}inner_ksp_type"] = "preonly"
            opts[f"{ksp_c.getOptionsPrefix()}inner_pc_type"] = "ilu"
            opts[f"{ksp_c.getOptionsPrefix()}inner_pc_factor_levels"] = 0
            opts[f"{ksp_c.getOptionsPrefix()}inner_pc_factor_fill"] = 2.0
            opts[f"{ksp_c.getOptionsPrefix()}upper_ksp_type"] = "preonly"
            opts[f"{ksp_c.getOptionsPrefix()}upper_pc_type"] = "ilu"
            opts[f"{ksp_c.getOptionsPrefix()}upper_pc_factor_levels"] = 0
            opts[f"{ksp_c.getOptionsPrefix()}upper_pc_factor_fill"] = 2.0

            # for optk, optv in solver_params.AMG_TYPES[args.amg_type].items():
            #     opts[f"{ksp_c.getOptionsPrefix()}{optk}"] = optv

            ksp_u.setFromOptions()
            ksp_c.setFromOptions()
            snes.getKSP().setFromOptions()

            problem = solvers.NonlinearPDE_SNESProblem(F, J, [u_0, u_1, c, cbar], bcs, P=P)
            snes.setFunction(problem.F_block, Fvec)
            snes.setJacobian(problem.J_block, J=Jmat, P=Pmat)
            snes.setFromOptions()
            snes.view()

            x = fem.petsc.create_vector_block(F)
            x.set(0.0)
            PETSc.Sys.Print("Solving SNES problem..")
            PETSc.Log().begin()
            t0 = time.time()
            snes.solve(None, x)
            t1 = time.time()
            PETSc.Sys.Print(f"SNES converged reason: {snes.getConvergedReason()}")
            PETSc.Log().view(log_viewer)
            if comm_rank == 0 and args.plot:
                fig, ax = plt.subplots()
                ax.semilogy(snes.getKSP().getConvergenceHistory(), 'x-')
                ax.set_box_aspect(1)
                plt.tight_layout()
                plt.savefig(convergence_history, bbox_inches="tight")
            snes.destroy()
            Jmat.destroy(), Fvec.destroy()
            x.destroy()
            Pmat.destroy()

        else:
            raise ValueError("Unknown solver type!")

        c0.x.array[:] = c.x.array
        cvtx.write(t)
        I_left = comm.allreduce(fem.assemble_scalar(fem.form(inner(kappa_elec * (phi_ref) * L_ref ** (tdim-2) * grad(u_0), n) * ds(markers.left), entity_maps=entity_maps)), op=MPI.SUM)
        I_right = comm.allreduce(fem.assemble_scalar(fem.form(inner(kappa_pos_am * (phi_ref) * L_ref ** (tdim-2) * grad(u_1), n) * ds(markers.right), entity_maps=entity_maps)), op=MPI.SUM)
        I_interface = comm.allreduce(fem.assemble_scalar(fem.form(inner(faraday_const * D * (c_ref) * L_ref ** (tdim-2) * grad(c(r_res)), n_r) * dInterface, entity_maps=entity_maps)), op=MPI.SUM)
    cvtx.close()

    time_elapsed = timeit.default_timer() - start_time

    resistance = args.voltage / (np.abs(I_left) * A0)
    metadata = {
        "I left [A]": I_left,
        "I interface [A]": I_interface,
        "I right [A]": I_right,
        "resistance [ohm.cm2]": resistance,
        "time elapsed [s]": time_elapsed,
        "solve time [s]": t1 - t0,
        "L ref [m]": ref["L"],
        "c ref [mol/m3]": ref["c"],
        "phi ref [V]": ref["phi"],
        "t ref [s]": ref["t"],
        "min time step [s]": args.dt * ref["t"],
        "Positive Wa": args.Wa_p,
        "Kr": args.kr,
        "polynomial approximation order (p)": args.p,
        "penalty parameter (gamma)": args.gamma,
        "kinetics": args.kinetics,
        "dofs": n_dofs,
        "n_procs": comm.Get_size(),
        "sim_date": datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    }
    if comm_rank == 0:
        utils.print_dict(metadata, padding=50)
        with open(simulation_metafile, "w", encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        PETSc.Sys.Print(f"Saved results files in {results_dir}")
        PETSc.Sys.Print(f"Wrote log summary to {log_datafile}")
        PETSc.Sys.Print(f"Time elapsed: {time_elapsed:3.5f}s")

    # interpolate
    V = fem.functionspace(domain, ("DG", 1))
    u = fem.Function(V)
    u.interpolate(u_0, cells1=submesh_electrolyte_to_mesh, cells0=np.arange(len(submesh_electrolyte_to_mesh)))
    u.interpolate(u_1, cells1=submesh_positive_am_to_mesh, cells0=np.arange(len(submesh_positive_am_to_mesh)))
    u.x.scatter_forward()

    with io.VTXWriter(comm, output_potential_file, [u], engine="BP5") as vtx:
        vtx.write(0)

    with io.VTXWriter(comm, elec_potential_file, [u_0], engine="BP5") as vtx:
        vtx.write(0)

    with io.VTXWriter(comm, positive_am_potential_file, [u_1], engine="BP5") as vtx:
        vtx.write(0)

    if args.plot:
        n_points = 1000
        if comm_rank == 0:
            all_vals = np.zeros((n_points, 4))
        tol = 1e-8  # Avoid hitting the outside of the domain

        z = np.linspace(tol, 1 - tol, n_points)
        points = np.zeros((3, n_points))
        points[2] = z

        # obtain concentration values to plot
        cells = []
        points_on_proc = []
        bb_trees = bb_tree(submesh_positive_am, submesh_positive_am.topology.dim)
        # Find cells whose bounding-box collide with the the points
        cell_candidates = compute_collisions_points(bb_trees, points.T)
        # Choose one of the cells that contains the point
        colliding_cells = compute_colliding_cells(submesh_positive_am, cell_candidates, points.T)

        # obtain potential values to plot
        cells_d = []
        points_on_proc_d = []
        bb_trees_d = bb_tree(domain, domain.topology.dim)
        # Find cells whose bounding-box collide with the the points
        cell_candidates_d = compute_collisions_points(bb_trees_d, points.T)
        # Choose one of the cells that contains the point
        colliding_cells_d = compute_colliding_cells(domain, cell_candidates_d, points.T)

        for i in range(n_points):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(points.T[i])
                cells.append(colliding_cells.links(i)[0])

            if len(colliding_cells_d.links(i)) > 0:
                points_on_proc_d.append(points.T[i])
                cells_d.append(colliding_cells_d.links(i)[0])

        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        points_on_proc_d = np.array(points_on_proc_d, dtype=np.float64)
        c_values_mid = c.eval(points_on_proc, cells)
        if np.all(c_values_mid.shape):
            try:
                c_plot_vals = np.hstack((points_on_proc, c_values_mid))
            except ValueError:
                c_plot_vals = np.empty((0, 4))
        else:
            c_plot_vals = np.empty((0, 4))

        u_values_mid = u.eval(points_on_proc_d, cells_d)
        if np.all(u_values_mid.shape):
            try:
                u_plot_vals = np.hstack((points_on_proc_d, u_values_mid))
            except ValueError:
                u_plot_vals = np.empty((0, 4))
        else:
            u_plot_vals = np.empty((0, 4))

        if comm_rank != 0:
            req = comm.send(c_plot_vals, dest=0, tag=11)
            req2 = comm.send(u_plot_vals, dest=0, tag=13)

        if comm_rank == 0:
            all_c_vals = c_plot_vals
            all_u_vals = u_plot_vals
            for rank in range(1, comm_size):
                addtnl_c = comm.recv(source=rank, tag=11)
                all_c_vals = np.vstack((all_c_vals, addtnl_c))

                addtnl_u = comm.recv(source=rank, tag=13)
                all_u_vals = np.vstack((all_u_vals, addtnl_u))

            c_vals = all_c_vals[all_c_vals[:, 2].argsort()]
            u_vals = all_u_vals[all_u_vals[:, 2].argsort()]

            fig, ax = plt.subplots()
            ax.plot(c_vals[:, 2], c_vals[:, 3], 'k', label=r'0.5$L_x$,0.5$L_y$', linewidth=1)
            ax.grid(True)
            ax.legend()
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_box_aspect(1)
            ax.set_ylabel(r'$\hat{c}$', rotation=90, labelpad=0, fontsize='xx-large')
            ax.set_xlabel(r'$\hat{x}$')
            ax.set_title(r'$\mathrm{Wa}$ = ' + f'{args.Wa_p}' + ',' + r'$\frac{\kappa}{\sigma}$ = ' + f'{args.kr}')
            plt.tight_layout()
            plt.savefig(concentration_plot_file)

            fig, ax = plt.subplots()
            ax.plot(u_vals[:, 2], u_vals[:, 3], 'k', label=r'0.5$L_x$,0.5$L_y$', linewidth=1)
            ax.grid(True)
            ax.legend()
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_box_aspect(1)
            ax.set_ylabel(r'$\hat{\phi}$', rotation=90, labelpad=0, fontsize='xx-large')
            ax.set_xlabel(r'$\hat{x}$')
            ax.set_title(r'$\mathrm{Wa}$ = ' + f'{args.Wa_p}' + ',' + r'$\frac{\kappa}{\sigma}$ = ' + f'{args.kr}')
            plt.tight_layout()
            plt.savefig(potential_plot_file)
