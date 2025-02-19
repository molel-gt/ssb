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
import scifem
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
logging.basicConfig(level=logging.INFO)

R = 8.314
T = 298
faraday_const = 96485
kappa_pos_am = 0.1
kinetics = ('linear', 'tafel', 'butler_volmer')
galvanostatic = "galvanostatic"
potentiostatic = "potentiostatic"
modes = (galvanostatic, potentiostatic)
micron = 1e-6
V_UCO = 5.0  # upper cutoff voltage
c_max = 35000
directions = {'x': 0, 'y': 1, 'z': 2}


log.set_log_level(dolfinx.log.LogLevel.WARNING)


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

    el = basix.ufl.element(basix.ElementFamily.P, cell_type, degree, basix.LagrangeVariant.gll_isaac, dtype=dolfinx.default_real_type)
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
    parser.add_argument("--D", help="Diffusivity [m2/s]", nargs='?', const=1, default=1e-14, type=float)
    parser.add_argument("--kr", help="ratio of ionic to electronic conductivity", nargs='?', const=1, default=1, type=float)
    parser.add_argument("--gamma", help="interior penalty parameter", nargs='?', const=1, default=15, type=float)
    parser.add_argument("-p_c", "--p_concentration", help="polynomial approximation order for concentration field", nargs='?', const=1, default=4, type=int)
    parser.add_argument("-p_u", "--p_potential", help="polynomial approximation order for potential field", nargs='?', const=1, default=2, type=int)
    parser.add_argument("-cell_type", "--cell_type", help="cell type to use", nargs='?', const=1, default="tetrahedron", type=str)
    parser.add_argument("-dt", "--dt", help="minimum normalized time step", nargs='?', const=1, default=2e-7, type=float)
    parser.add_argument("-sim_time", "--sim_time", help="simulation time in seconds", nargs='?', const=1, default=15, type=float)
    parser.add_argument("-cycle_mode", "--cycle_mode", help="mode of cycling", nargs='?', const=1, default="galvanostatic", type=str)
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
    parser.add_argument("-C_rate", "--C_rate", help="cycling rate", nargs='?', const=1, default=0.1, type=float)
    parser.add_argument("--compute_distribution", help="compute current distribution stats", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    if args.cycle_mode not in modes:
        raise ValueError(f"Only {modes.__repr__()} allowed")

    PETSc.Sys.Print("************************************** CYCLING PARAMETERS SUMMARY *************************************")
    PETSc.Sys.Print("cycle mode                                             :", args.cycle_mode)
    if args.cycle_mode == potentiostatic:
        PETSc.Sys.Print("Voltage [V]                                            :", args.voltage)
    elif args.cycle_mode == galvanostatic:
        PETSc.Sys.Print("C-rate                                                 :", args.C_rate)
    PETSc.Sys.Print("simulation time [s]                                    :", args.sim_time)
    PETSc.Sys.Print("Positive electrode Wa                                  :", args.Wa_p)
    PETSc.Sys.Print("Conductivity ratio (Kr)                                :", args.kr)
    PETSc.Sys.Print("Lithium diffusivity in positive active material [m2/s] :", args.D)
    PETSc.Sys.Print("Kinetics                                               :", args.kinetics)
    PETSc.Sys.Print("*******************************************************************************************************")
    PETSc.Sys.Print("************************************** SOLVER PARAMETERS **********************************************")
    PETSc.Sys.Print("interior penalty parameter (gamma)                     :", args.gamma)
    PETSc.Sys.Print("solve improved guesss                                  :", args.improved_guess)
    PETSc.Sys.Print("minimum dt [normalized]                                :", args.dt)
    PETSc.Sys.Print("*******************************************************************************************************")

    start_time = timeit.default_timer()
    voltage = args.voltage
    Wa_n = args.Wa_n
    Wa_p = args.Wa_p
    kappa_elec = args.kr * kappa_pos_am
    D = args.D

    markers = commons.Markers()
    solver_types = SolverTypes()
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    cell_type = getattr(basix.CellType, args.cell_type)

    dimensions = utils.extract_dimensions_from_meshfolder(args.mesh_folder)
    LX, LY, LZ = [float(vv) * micron for vv in dimensions.split("-")]

    L_ref = get_Lref([LX, LY, LZ], args.transport_direction)

    # reference values
    t_ref = L_ref ** 2 / D
    phi_ref = V_UCO
    # c_ref = c_max
    c_ref = kappa_pos_am * phi_ref / (faraday_const * D)
    ref = {"t": t_ref, "phi": phi_ref, "c": c_ref, "L": L_ref}
    R_p_ref = 10e-6  # characteristic diffusion length

    TIME = args.dt#args.sim_time / t_ref

    # exchange current densities
    i0_n = kappa_elec * R * T / (Wa_n * faraday_const * L_ref)
    i0_p = kappa_elec * R * T / (Wa_p * faraday_const * L_ref)

    thiele = R_p_ref * i0_p * V_UCO / (R * T * D * c_max)

    soc_init = 0.75 * c_max / c_ref

    output_meshfile = os.path.join(args.mesh_folder, "mesh.msh")
    results_dir = os.path.join(args.mesh_folder, args.cycle_mode, args.kinetics, str(Wa_n) + "-" + str(Wa_p) + "-" + str(args.kr), f'{args.C_rate}C',f'{args.p_potential}-{args.p_concentration}', str(args.gamma), str(comm.Get_size()))
    utils.make_dir_if_missing(results_dir)
    output_potential_file = os.path.join(results_dir, "potential.bp")
    elec_potential_file = os.path.join(results_dir, "electrolyte_potential.bp")
    positive_am_potential_file = os.path.join(results_dir, "positive_am_potential.bp")
    current_file = os.path.join(results_dir, "current.bp")
    concentration_file = os.path.join(results_dir, "concentration.bp")
    potential_plot_file = os.path.join(results_dir, "potential.eps")
    concentration_plot_file = os.path.join(results_dir, "concentration.eps")
    simulation_metafile = os.path.join(results_dir, "simulation.json")
    stats_metadata_file = os.path.join(results_dir, "stats.json")
    convergence_history = os.path.join(results_dir, "convergence.eps")
    se_am_frequency_plot = os.path.join(results_dir, "se_am_frequency.eps")
    se_am_cdf_plot = os.path.join(results_dir, "se_am_cdf.eps")
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
        domain, tdim, ct.find(markers.electrolyte)
    )[0:3]
    submesh_positive_am, submesh_positive_am_to_mesh, t_v_map = mesh.create_submesh(
        domain, tdim, ct.find(markers.positive_am)
    )[0:3]
    parent_to_sub_electrolyte = np.full(num_facets_local, -1, dtype=np.int32)
    parent_to_sub_electrolyte[submesh_electrolyte_to_mesh] = np.arange(len(submesh_electrolyte_to_mesh), dtype=np.int32)
    parent_to_sub_positive_am = np.full(num_facets_local, -1, dtype=np.int32)
    parent_to_sub_positive_am[submesh_positive_am_to_mesh] = np.arange(len(submesh_positive_am_to_mesh), dtype=np.int32)

    ft_electrolyte = mesh_utils.transfer_meshtags(domain, submesh_electrolyte, submesh_electrolyte_to_mesh, ft)
    ft_positive_am = mesh_utils.transfer_meshtags(domain, submesh_positive_am, submesh_positive_am_to_mesh, ft)


    # Hack, as we use one-sided restrictions, pad dS integral with the same entity from the same cell on both sides
    domain.topology.create_connectivity(fdim, tdim)
    f_to_c = domain.topology.connectivity(fdim, tdim)

    for facet in ft.find(markers.electrolyte_v_positive_am):
        cells = f_to_c.links(facet)
        assert len(cells) == 2
        b_map = parent_to_sub_electrolyte[cells]
        t_map = parent_to_sub_positive_am[cells]
        parent_to_sub_electrolyte[cells] = max(b_map)
        parent_to_sub_positive_am[cells] = max(t_map)

    entity_maps = {submesh_electrolyte: parent_to_sub_electrolyte, submesh_positive_am: parent_to_sub_positive_am}

    u_0, F_00, m_to_elec = define_interior_eq(domain, args.p_potential, submesh_electrolyte, submesh_electrolyte_to_mesh, 0.0, kappa_elec, cell_type)
    u_1, F_11, m_to_pos_am = define_interior_eq(domain, args.p_potential, submesh_positive_am, submesh_positive_am_to_mesh, 0.0, kappa_pos_am, cell_type)
    u_0.name = "u_b"
    u_1.name = "u_t"

    # initial guess
    u_0.interpolate(lambda x: x[0]-x[0])#x[directions[args.transport_direction.lower()]]*0.95)
    u_1.interpolate(lambda x: 0.5 + x[0]-x[0])#1.05*x[directions[args.transport_direction.lower()]]/1.1)

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
    dx = ufl.Measure('dx', domain=domain, subdomain_data=ct)
    dx_r = ufl.Measure('dx', domain=domain, subdomain_data=ct, subdomain_id=markers.positive_am)
    dx_c = ufl.Measure('dx', domain=submesh_positive_am)
    ds = ufl.Measure('ds', domain=domain, subdomain_data=ft)
    ds_c = ufl.Measure('ds', domain=submesh_positive_am, subdomain_data=ft_positive_am)

    vol_pos_am = comm.allreduce(fem.assemble_scalar(fem.form(1 * dx(markers.positive_am), entity_maps=entity_maps)), op=MPI.SUM) * L_ref ** 3
    I_tot_ = utils.get_c_rate_current(c_max, args.C_rate, vol_pos_am)
    l_res = "-"
    r_res = "+"
    V0 = u_0.function_space
    V1 = u_1.function_space

    v_0 = ufl.TestFunction(V0)
    v_1 = ufl.TestFunction(V1)

    v_l = v_0(l_res)
    v_r = v_1(r_res)
    u_l = u_0(l_res)
    u_r = u_1(r_res)

    n = ufl.FacetNormal(domain)
    n_0 = ufl.FacetNormal(submesh_electrolyte)
    n_1 = ufl.FacetNormal(submesh_positive_am)
    n_l = n(l_res)
    n_r = n(r_res)
    cd = ufl.CellDiameter(domain)
    h_1 = ufl.CellDiameter(submesh_positive_am)
    h_l = cd(l_res)
    h_r = cd(r_res)

    # concentration problem
    dt = fem.Constant(submesh_positive_am, args.dt)
    el = basix.ufl.element(basix.ElementFamily.P, cell_type, args.p_concentration, basix.LagrangeVariant.gll_isaac, dtype=dolfinx.default_real_type)
    VC = fem.functionspace(submesh_positive_am, el)

    c, q = fem.Function(VC), ufl.TestFunction(VC)
    c0 = fem.Function(VC)
    u_int = fem.Function(VC)

    c0.interpolate(lambda x: x[directions[args.transport_direction.lower()]] - x[directions[args.transport_direction.lower()]] + soc_init)
    # c.interpolate(c0)#lambda x: soc_init * (1 - np.exp(-x[directions[args.transport_direction.lower()]])))

    q_r = ufl.TestFunction(c.function_space)(r_res)
    q_l = ufl.TestFunction(c.function_space)(l_res)
    c_r = c(r_res)

    jump_u = surface_overpotential(kappa_pos_am, u_r, n_r, i0_p, kinetics_type=args.kinetics, ref=ref) + ocv_chen2020(c(r_res), cmax=c_max/c_ref)/phi_ref

    # for galvanostatic mode
    # left facets submesh
    submesh_facets_left, submesh_facets_left_to_mesh = mesh.create_submesh(
        domain, fdim, ft.find(markers.left))[:2]
    num_facets_local = (
        domain.topology.index_map(fdim).size_local + domain.topology.index_map(fdim).num_ghosts
    )
    parent_to_facets_left = np.full(num_facets_local, -1, dtype=np.int32)
    parent_to_facets_left[submesh_facets_left_to_mesh] = np.arange(len(submesh_facets_left_to_mesh), dtype=np.int32)
    entity_maps[submesh_facets_left] = parent_to_facets_left
    # right facets submesh
    submesh_facets_right, submesh_facets_right_to_mesh = mesh.create_submesh(
        domain, fdim, ft.find(markers.right))[:2]
    num_facets_local = (
        domain.topology.index_map(fdim).size_local + domain.topology.index_map(fdim).num_ghosts
    )
    parent_to_facets_right = np.full(num_facets_local, -1, dtype=np.int32)
    parent_to_facets_right[submesh_facets_right_to_mesh] = np.arange(len(submesh_facets_right_to_mesh), dtype=np.int32)
    entity_maps[submesh_facets_right] = parent_to_facets_right

    all_facets = mesh_utils.compute_cell_boundary_facets(domain, ct, [markers.positive_am, markers.electrolyte])
    right_facets = mesh_utils.compute_interface_cell_boundary_facets(domain, ct, ft, markers.positive_am, markers.right)
    left_facets = mesh_utils.compute_interface_cell_boundary_facets(domain, ct, ft, markers.electrolyte, markers.left)
    minus_right_facets = utils.delete_numpy_rows(all_facets, right_facets)
    right_bndry_facets = np.array(right_facets).flatten()
    left_bndry_facets = np.array(left_facets).flatten()
    ds_f = ufl.Measure("ds", subdomain_data=[(1, minus_right_facets.flatten()), (2, left_bndry_facets), (3, right_bndry_facets)], domain=domain)
    A_left_tilde = comm.allreduce(fem.assemble_scalar(fem.form(1 * ds(markers.left))), op=MPI.SUM)
    A_left = A_left_tilde * (L_ref ** 2)

    A_right_tilde = comm.allreduce(fem.assemble_scalar(fem.form(1 * ds(markers.right))), op=MPI.SUM)
    A_right = A_right_tilde * (L_ref ** 2)

    A_se_am_tilde = comm.allreduce(fem.assemble_scalar(fem.form(1 * ds_c(markers.electrolyte_v_positive_am))), op=MPI.SUM)
    A_se_am = A_se_am_tilde * (L_ref ** 2)
    A_se_am_to_vol_am = A_se_am / vol_pos_am
    PETSc.Sys.Print("Area Left [m2]                        :", f"{A_left:.0e}")
    PETSc.Sys.Print("Area Right [m2]                       :", f"{A_right:.0e}")
    PETSc.Sys.Print("Area SE/AM [m2]                       :", f"{A_se_am:.0e}")
    PETSc.Sys.Print("SE/AM area to cross-section area      :", f"{A_se_am/A_right:,.0f}")
    PETSc.Sys.Print("SE/AM area to volume ratio            :", f"{A_se_am_to_vol_am:,.0f}")

    R_right = scifem.create_real_functionspace(submesh_facets_right)
    el_V_r = basix.ufl.element(basix.ElementFamily.P, basix.CellType.triangle, args.p_potential, basix.LagrangeVariant.gll_isaac, dtype=dolfinx.default_real_type)
    V_r = fem.functionspace(submesh_facets_right, el_V_r)
    lmbda, mu = fem.Function(V_r), ufl.TestFunction(V_r)

    V_cell, w = fem.Function(R_right), ufl.TestFunction(R_right)

    I_tot = fem.Constant(submesh_facets_right, PETSc.ScalarType(-I_tot_))
    i_sup = np.abs(I_tot.value) / A_right
    h = ufl.CellDiameter(submesh_facets_right)
    gamma = fem.Constant(domain, PETSc.ScalarType(args.gamma))
    alpha = 1e-6 # preturbation penalty

    F_0 = (
        - 0.5 * mixed_term(kappa_elec * u_l + kappa_pos_am * u_r, v_l, n_l) * dInterface
        - 0.5 * mixed_term(0.5 * (kappa_elec + kappa_pos_am) * v_l, (u_r - u_l - jump_u), n_l) * dInterface
    )

    F_1 = (
        + 0.5 * mixed_term(kappa_elec * u_l + kappa_pos_am * u_r, v_r, n_l) * dInterface
        - 0.5 * mixed_term(0.5 * (kappa_elec + kappa_pos_am) * v_r, (u_r - u_l - jump_u), n_l) * dInterface
    )
    F_0 += -2 * gamma / (h_l + h_r) * (u_r - u_l - jump_u) * v_l * dInterface
    F_1 += +2 * gamma / (h_l + h_r) * (u_r - u_l - jump_u) * v_r * dInterface

    if args.cycle_mode == galvanostatic:
        F_1 += - v_1 * lmbda * ds_f(3)
        F_1a = (V_cell - u_1) * mu * ds_f(3)
        F_1b = w * (I_tot/(A_right_tilde * L_ref * phi_ref) + lmbda) * ds_f(3) - 1e-8/h * (V_cell - u_1) * w * ds_f(3)
        # F_1 += -(V_cell - u_1) * lmbda * ds_f(3)

    F_0 += F_00
    F_1 += F_11

    F_2 = (c - c0)/dt * q * dx_r + inner(ufl.grad(c), ufl.grad(q)) * dx_r
    # F_2 += -inner(kappa_pos_am * phi_ref/(D * faraday_const * c_ref) * grad(u_r), n_r) * q_r * dInterface
    F_2 += -inner(grad(u_r), n_r) * q_r * dInterface
    F_2 += alpha * h_r * inner(inner(-grad(u_r) + grad(c_r), n_r), inner(grad(q_r), n_r)) * dInterface

    u_left = fem.Function(V0)
    u_left.x.array[:] = 0/phi_ref
    submesh_electrolyte.topology.create_connectivity(
        submesh_electrolyte.topology.dim - 1, submesh_electrolyte.topology.dim
    )
    bc_left = fem.dirichletbc(
        u_left, fem.locate_dofs_topological(u_0.function_space, fdim, ft_electrolyte.find(markers.left))
    )

    u_right = fem.Function(V1)
    u_right.x.array[:] = args.voltage/phi_ref
    submesh_positive_am.topology.create_connectivity(
        submesh_positive_am.topology.dim - 1, submesh_positive_am.topology.dim
    )
    bc_right = fem.dirichletbc(
        u_right, fem.locate_dofs_topological(u_1.function_space, fdim, ft_positive_am.find(markers.right))
    )

    bcs = [bc_left]
    if args.cycle_mode == potentiostatic:
        bcs = [bc_left, bc_right]

    if args.cycle_mode == galvanostatic:
        jac00 = ufl.derivative(F_0, u_0)
        jac01 = ufl.derivative(F_0, u_1)
        jac02 = ufl.derivative(F_0, lmbda)
        jac03 = ufl.derivative(F_0, V_cell)
        jac04 = ufl.derivative(F_0, c)

        jac10 = ufl.derivative(F_1, u_0)
        jac11 = ufl.derivative(F_1, u_1)
        jac12 = ufl.derivative(F_1, lmbda)
        jac13 = ufl.derivative(F_1, V_cell)
        jac14 = ufl.derivative(F_1, c)

        jac20 = ufl.derivative(F_1a, u_0)
        jac21 = ufl.derivative(F_1a, u_1)
        jac22 = ufl.derivative(F_1a, lmbda)
        jac23 = ufl.derivative(F_1a, V_cell)
        jac24 = ufl.derivative(F_1a, c)

        jac30 = ufl.derivative(F_1b, u_0)
        jac31 = ufl.derivative(F_1b, u_1)
        jac32 = ufl.derivative(F_1b, lmbda)
        jac33 = ufl.derivative(F_1b, V_cell)
        jac34 = ufl.derivative(F_1b, c)

        jac40 = ufl.derivative(F_2, u_0)
        jac41 = ufl.derivative(F_2, u_1)
        jac42 = ufl.derivative(F_2, lmbda)
        jac43 = ufl.derivative(F_2, V_cell)
        jac44 = ufl.derivative(F_2, c)

        J00 = fem.form(jac00, entity_maps=entity_maps)
        J01 = fem.form(jac01, entity_maps=entity_maps)
        J02 = fem.form(jac02, entity_maps=entity_maps)
        J03 = fem.form(jac03, entity_maps=entity_maps)
        J04 = fem.form(jac04, entity_maps=entity_maps)

        J10 = fem.form(jac10, entity_maps=entity_maps)
        J11 = fem.form(jac11, entity_maps=entity_maps)
        J12 = fem.form(jac12, entity_maps=entity_maps)
        J13 = fem.form(jac13, entity_maps=entity_maps)
        J14 = fem.form(jac14, entity_maps=entity_maps)

        J20 = fem.form(jac20, entity_maps=entity_maps)
        J21 = fem.form(jac21, entity_maps=entity_maps)
        J22 = fem.form(jac22, entity_maps=entity_maps)
        J23 = fem.form(jac23, entity_maps=entity_maps)
        J24 = fem.form(jac24, entity_maps=entity_maps)

        J30 = fem.form(jac30, entity_maps=entity_maps)
        J31 = fem.form(jac31, entity_maps=entity_maps)
        J32 = fem.form(jac32, entity_maps=entity_maps)
        J33 = fem.form(jac33, entity_maps=entity_maps)
        J34 = fem.form(jac34, entity_maps=entity_maps)

        J40 = fem.form(jac40, entity_maps=entity_maps)
        J41 = fem.form(jac41, entity_maps=entity_maps)
        J42 = fem.form(jac42, entity_maps=entity_maps)
        J43 = fem.form(jac43, entity_maps=entity_maps)
        J44 = fem.form(jac44, entity_maps=entity_maps)

        J = [
            [J00, J01, J02, J03, J04],
            [J10, J11, J12, J13, J14],
            [J20, J21, J22, J23, J24],
            [J30, J31, J32, J33, J34],
            [J40, J41, J42, J43, J44],
        ]

    elif args.cycle_mode == potentiostatic:
        jac00 = ufl.derivative(F_0, u_0)
        jac01 = ufl.derivative(F_0, u_1)
        jac02 = ufl.derivative(F_0, c)

        jac10 = ufl.derivative(F_1, u_0)
        jac11 = ufl.derivative(F_1, u_1)
        jac12 = ufl.derivative(F_1, c)

        jac20 = ufl.derivative(F_2, u_0)
        jac21 = ufl.derivative(F_2, u_1)
        jac22 = ufl.derivative(F_2, c)

        J00 = fem.form(jac00, entity_maps=entity_maps)
        J01 = fem.form(jac01, entity_maps=entity_maps)
        J02 = fem.form(jac02, entity_maps=entity_maps)

        J10 = fem.form(jac10, entity_maps=entity_maps)
        J11 = fem.form(jac11, entity_maps=entity_maps)
        J12 = fem.form(jac12, entity_maps=entity_maps)

        J20 = fem.form(jac20, entity_maps=entity_maps)
        J21 = fem.form(jac21, entity_maps=entity_maps)
        J22 = fem.form(jac22, entity_maps=entity_maps)

        J = [
            [J00, J01, J02,],
            [J10, J11, J12,],
            [J20, J21, J22,],
        ]

    V0_map = V0.dofmap.index_map
    V1_map = V1.dofmap.index_map
    VC_map = VC.dofmap.index_map
    V_r_map = V_r.dofmap.index_map
    R_right_map = R_right.dofmap.index_map
    V0_dofmap = V0.dofmap
    V1_dofmap = V1.dofmap
    VC_dofmap = VC.dofmap
    V_r_dofmap = V_r.dofmap
    R_right_dofmap = R_right.dofmap

    ###################### sparsity structure ##################################
    if args.plot_sparsity:
        J_full = fem.petsc.assemble_matrix_block(J)
        J_full.assemble()

        # viewer = PETSc.Viewer().createDraw(size=(1200, 1200))
        # viewer(J_full)
        ai, aj, av = J_full.getValuesCSR()
        Asp = scipy.sparse.csr_matrix((av, aj, ai))
        fig, ax = plt.subplots()
        ax.spy(Asp, markersize=0.25)
        ax.grid()
        _l0 = V0_map.size_local
        _l1 = _l0 + V1_map.size_local
        ax.axvline(x=_l0, color='red', linewidth=0.5)
        ax.axhline(y=_l0, color='red', linewidth=0.5)
        ax.axvline(x=_l1, color='red', linewidth=0.5)
        ax.axhline(y=_l1, color='red', linewidth=0.5)
        ax.set_box_aspect(1);
        plt.savefig(os.path.join(results_dir, "jacobian-sparsity.png"), bbox_inches='tight')#, transparent=True)
    ############################################################################
    if args.cycle_mode == galvanostatic:
        F = [
            fem.form(F_0, entity_maps=entity_maps),
            fem.form(F_1, entity_maps=entity_maps),
            fem.form(F_1a, entity_maps=entity_maps),
            fem.form(F_1b, entity_maps=entity_maps),
            fem.form(F_2, entity_maps=entity_maps),
        ]

        n_dofs = V0_map.size_global*V0.dofmap.index_map_bs + V1_map.size_global*V1.dofmap.index_map_bs + VC_map.size_global*VC.dofmap.index_map_bs +\
                V_r_map.size_global*V_r.dofmap.index_map_bs + R_right_map.size_global*R_right.dofmap.index_map_bs
    elif args.cycle_mode == potentiostatic:
        F = [
            fem.form(F_0, entity_maps=entity_maps),
            fem.form(F_1, entity_maps=entity_maps),
            fem.form(F_2, entity_maps=entity_maps),
        ]
        n_dofs = V0_map.size_global*V0.dofmap.index_map_bs + V1_map.size_global*V1.dofmap.index_map_bs + VC_map.size_global*VC.dofmap.index_map_bs


    t = 0
    cvtx = io.VTXWriter(comm, concentration_file, [c], engine="BP5")
    PETSc.Sys.Print(f"Setting up problem Wa: {args.Wa_p}, Kr: {args.kr}, #DoFs: {n_dofs:,}, nprocs: {comm.Get_size()}")
    P = J

    log_viewer = PETSc.Viewer().STDOUT()
    log_viewer.setFileName(log_datafile)
    ########################################################################################################################################
    ## solve initial potential distribution at t = 0
    if args.improved_guess:
        PETSc.Sys.Print("************Begin Solve for t = 0 Potential Distribution*******************")
        if args.cycle_mode == galvanostatic:
            n_dofs_t0 = V0_map.size_global*V0.dofmap.index_map_bs + V1_map.size_global*V1.dofmap.index_map_bs +\
                V_r_map.size_global*V_r.dofmap.index_map_bs + R_right_map.size_global*R_right.dofmap.index_map_bs
            F2D = F[:4]
            J2D = [j2d[:4] for j2d in J[:4]]
        elif args.cycle_mode == potentiostatic:
            n_dofs_t0 = V0_map.size_global*V0.dofmap.index_map_bs + V1_map.size_global*V1.dofmap.index_map_bs
            F2D = F[:2]
            J2D = [j2d[:2] for j2d in J[:2]]
        opts = PETSc.Options()
        Jmat2d = fem.petsc.create_matrix_block(J2D)
        Fvec2d = fem.petsc.create_vector_block(F2D)
        snes = PETSc.SNES().create(comm)
        snes.setType('newtonls')
        snes.setTolerances(rtol=5e-4, max_it=200)
        snes.setMonitor(lambda _, it, residual: PETSc.Sys.Print("it:", it, "res:", residual))
        snes.getKSP().setType(PETSc.KSP.Type.FGMRES)
        snes.getKSP().getPC().setType(PETSc.PC.Type.ILU)
        snes.getKSP().setOptionsPrefix("snes_")
        snes.getKSP().setOperators(Jmat2d, Jmat2d)
        snes.getKSP().setTolerances(rtol=1e-7)
        snes.setErrorIfNotConverged(True)
        snes.getKSP().setErrorIfNotConverged(True)
        snes.getKSP().setConvergenceHistory()
        for kopt, vopt in solver_params.LINESEARCH.items():
                opts[kopt] = vopt
        opts[f"{snes.getKSP().getOptionsPrefix()}pc_factor_levels"] = 0
        opts[f"{snes.getKSP().getOptionsPrefix()}pc_factor_fill"] = 2.0
        snes.getKSP().setFromOptions()
        snes.setFromOptions()
        snes.view()

        if args.cycle_mode == galvanostatic:
            _sol_vars = [u_0, u_1, lmbda, V_cell]
        elif args.cycle_mode == potentiostatic:
            _sol_vars = [u_0, u_1]

        problem_t0 = solvers.NonlinearPDE_SNESProblem(F2D, J2D, _sol_vars, bcs, P=J2D)
        snes.setFunction(problem_t0.F_block, Fvec2d)
        snes.setJacobian(problem_t0.J_block, J=Jmat2d, P=Jmat2d)
        x2d = fem.petsc.create_vector_block(F2D)
        x2d.set(0.0)
        t0 = time.time()
        snes.solve(None, x2d)
        t1 = time.time()
        snes.destroy()
        Jmat2d.destroy()
        Fvec2d.destroy()
        x2d.destroy()
        PETSc.Sys.Print(f"Finished computation of initial (t = 0) potential distribution!\nn_dofs: {n_dofs_t0}\nsolve time: {t1 - t0:.3f}s")
        PETSc.Sys.Print("************Solve for Improved Guess for Concentration Distribution*******************")
        n_dofs_c = VC_map.size_global*VC.dofmap.index_map_bs
        u_int.interpolate(u_1)
        F_c = (c - c0)/dt * q * dx_c + inner(ufl.grad(c), ufl.grad(q)) * dx_c
        # F_2 += -inner(kappa_pos_am * phi_ref/(D * faraday_const * c_ref) * grad(u_int), n_1) * q * ds_c(markers.electrolyte_v_positive_am)
        F_c += -inner(grad(u_int), n_1) * q * ds_c(markers.electrolyte_v_positive_am)
        problem_c = fem.petsc.NonlinearProblem(F_c, c, bcs=[])
        solver = petsc_nls.NewtonSolver(comm, problem_c)
        solver.convergence_criterion = "residual"
        solver.maximum_iterations = 100
        solver.rtol = 1e-8

        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "cg"
        opts[f"{option_prefix}pc_type"] = args.amg_type
        for optk, optv in solver_params.AMG_TYPES[args.amg_type].items():
                opts[f"{option_prefix}{optk}"] = optv
        # opts[f"{option_prefix}pc_factor_levels"] = 0
        # opts[f"{option_prefix}pc_factor_fill"] = 2.0
        ksp.setFromOptions()
        t0 = time.time()
        n_iters, converged = solver.solve(c)
        t1 = time.time()
        PETSc.Sys.Print(f"Finished computation of improved guess of concentration distribution!\nn_dofs: {n_dofs_c}\nsolve time: {t1 - t0:.3f}s")
    ########################################################################################################################################
    if args.cycle_mode == galvanostatic:
        sol_vars = [u_0, u_1, lmbda, V_cell, c]
    elif args.cycle_mode == potentiostatic:
        sol_vars = [u_0, u_1, c]

    stats = []

    idx = 0
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
                soln_vars,
                bcs=bcs,
                max_iterations=1000,
                petsc_options=opts,
                )
            PETSc.Log().begin()
            t0 = time.time()
            solver.solve()
            t1 = time.time()
            PETSc.Log().view(log_viewer)
        elif args.solver_type == solver_types.nested_iterative:
            Jmat = fem.petsc.create_matrix_nest(J)
            nested_IS = Jmat.getNestISs()
            IS_u0 = nested_IS[0][0]
            IS_u1 = nested_IS[0][1]
            IS_u = IS_u0.sum(IS_u1)
            IS_c = nested_IS[0][2]
            Pmat = fem.petsc.create_matrix_nest(P)
            Fvec = fem.petsc.create_vector_nest(F)
            snes = PETSc.SNES().create(comm)
            snes.setType('newtonls')
            snes.setTolerances(rtol=1.0e-7, max_it=15)
            snes.getKSP().setType(PETSc.KSP.Type.FGMRES)
            snes.getKSP().setOptionsPrefix("snes_")
            snes.getKSP().setOperators(Jmat, Pmat)
            nullspace = PETSc.NullSpace().create(constant=True)
            PETSc.Mat.setNearNullSpace(Jmat, nullspace)
            snes.getKSP().setTolerances(rtol=1e-7)
            snes.setErrorIfNotConverged(True)
            snes.getKSP().setErrorIfNotConverged(True)
            snes.getKSP().setConvergenceHistory()
            snes.getKSP().getPC().setType("fieldsplit")
            IS_u0 = nested_IS[0][0]
            IS_u1 = nested_IS[0][1]
            IS_u = IS_u0.sum(IS_u1)
            IS_c = nested_IS[0][2]
            snes.getKSP().getPC().setFieldSplitIS(("u", IS_u), ("c", IS_c))
            opts = PETSc.Options()
            for kopt, vopt in solver_params.LINESEARCH.items():
                opts[kopt] = vopt

            opts[f"{snes.getKSP().getOptionsPrefix()}pc_fieldsplit_off_diag_use_amat"] = True
            opts[f"{snes.getKSP().getOptionsPrefix()}pc_fieldsplit_detect_saddle_point"] = True

            ksp_u, ksp_c = snes.getKSP().getPC().getFieldSplitSubKSP()

            snes.getKSP().getPC().setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
            snes.getKSP().getPC().setFieldSplitSchurPreType(PETSc.PC.SchurPreType.SELFP)
            snes.getKSP().getPC().setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)

            ksp_u.setType(PETSc.KSP.Type.FGMRES)
            ksp_u.getPC().setType(PETSc.PC.Type.JACOBI)
            opts[f"{ksp_u.getOptionsPrefix()}pc_jacobi_fixdiagonal"] = True
            ksp_u.setConvergenceHistory()
            ksp_c.setType(PETSc.KSP.Type.FGMRES)
            ksp_c.getPC().setType(args.amg_type)
            ksp_c.setConvergenceHistory()

            opts[f'{snes.getKSP().getOptionsPrefix()}ksp_gmres_restart'] = 75
            opts[f'{ksp_u.getOptionsPrefix()}ksp_gmres_restart'] = 75
            opts[f'{ksp_c.getOptionsPrefix()}ksp_gmres_restart'] = 75

            opts[f"{ksp_c.getOptionsPrefix()}mat_schur_complement_ainv_type"] = "lump"

            for optk, optv in solver_params.AMG_TYPES[args.amg_type].items():
                opts[f"{ksp_c.getOptionsPrefix()}{optk}"] = optv

            ksp_u.setFromOptions()
            ksp_c.setFromOptions()
            snes.getKSP().setFromOptions()

            problem = solvers.NonlinearPDE_SNESProblem(F, J, sol_vars, bcs, P=P)
            snes.setFunction(problem.F_nest, Fvec)
            snes.setJacobian(problem.J_nest, J=Jmat, P=Pmat)
            snes.setFromOptions()
            snes.view()

            x = fem.petsc.create_vector_nest(F)
            for x1_soln_pair in zip(x.getNestSubVecs(), (u_0, u_1, c)):
                x1_sub, soln_sub = x1_soln_pair
                soln_sub.x.petsc_vec.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
                )
                soln_sub.x.petsc_vec.copy(result=x1_sub)
                x1_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            x.set(0.0)
            # PETSc.Log().begin()
            t0 = time.time()
            snes.solve(None, x)
            t1 = time.time()
            PETSc.Sys.Print(f"SNES converged reason: {snes.getConvergedReason()}")
            # PETSc.Log().view(log_viewer)
            if comm_rank == 0 and args.plot:
                fig, ax = plt.subplots()
                ax.semilogy(snes.getKSP().getConvergenceHistory())
                ax.set_box_aspect(1)
                plt.tight_layout()
                plt.savefig(convergence_history, bbox_inches="tight")
            snes.destroy()
            Jmat.destroy(), Fvec.destroy()
            x.destroy()
            Pmat.destroy()
        elif args.solver_type == solver_types.block_iterative:
            opts.clear()
            Jmat = fem.petsc.create_matrix_nest(J)
            nested_IS = Jmat.getNestISs()
            if args.cycle_mode == galvanostatic:
                IS_u0 = nested_IS[0][0]
                IS_u1 = nested_IS[0][1]
                IS_l = nested_IS[0][2]
                IS_g = nested_IS[0][3]
                IS_c = nested_IS[0][4]
                IS_u = IS_u0.sum(IS_u1)
                IS_ulg = IS_u.sum(IS_l).sum(IS_g)
            elif args.cycle_mode == potentiostatic:
                IS_u0 = nested_IS[0][0]
                IS_u1 = nested_IS[0][1]
                IS_c = nested_IS[0][2]
                IS_ulg = IS_u0.sum(IS_u1)

            Jmat = fem.petsc.create_matrix_block(J)
            Pmat = fem.petsc.create_matrix_block(P)
            Fvec = fem.petsc.create_vector_block(F)
            snes = PETSc.SNES().create(comm)
            snes.setType('newtonls')
            snes.setTolerances(rtol=1.0e-7, max_it=100)
            snes.getKSP().setType(PETSc.KSP.Type.FGMRES)
            snes.getKSP().setOptionsPrefix("snes_")
            snes.getKSP().setOperators(Jmat, Pmat)
            nullspace = PETSc.NullSpace().create(constant=True)
            PETSc.Mat.setNearNullSpace(Jmat, nullspace)
            snes.getKSP().setTolerances(rtol=1e-7)
            snes.setErrorIfNotConverged(True)
            snes.getKSP().setErrorIfNotConverged(True)
            snes.getKSP().setConvergenceHistory()
            snes.getKSP().getPC().setType("fieldsplit")
            snes.getKSP().getPC().setFieldSplitIS(("u", IS_ulg), ("c", IS_c))
            opts = PETSc.Options()
            opts[f'{snes.getKSP().getOptionsPrefix()}ksp_gmres_restart'] = 100
            for kopt, vopt in solver_params.LINESEARCH.items():
                opts[kopt] = vopt

            opts['log_view'] = None

            opts[f"{snes.getKSP().getOptionsPrefix()}pc_fieldsplit_off_diag_use_amat"] = True
            opts[f"{snes.getKSP().getOptionsPrefix()}pc_fieldsplit_detect_saddle_point"] = True

            ksp_u, ksp_c = snes.getKSP().getPC().getFieldSplitSubKSP()

            snes.getKSP().getPC().setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
            snes.getKSP().getPC().setFieldSplitSchurPreType(PETSc.PC.SchurPreType.SELFP)
            snes.getKSP().getPC().setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)

            ksp_u.setType(PETSc.KSP.Type.FGMRES)
            ksp_u.getPC().setType(PETSc.PC.Type.ILU)
            ksp_u.setTolerances(rtol=1e-7, max_it=1000)
            opts[f"{ksp_u.getOptionsPrefix()}pc_factor_levels"] = 0
            opts[f"{ksp_u.getOptionsPrefix()}pc_factor_fill"] = 2.0

            ksp_c.setType(PETSc.KSP.Type.CG)
            ksp_c.getPC().setType(args.amg_type)
            ksp_c.setTolerances(rtol=1e-7, max_it=1000)

            opts[f"{ksp_c.getOptionsPrefix()}mat_schur_complement_ainv_type"] = "lump"
            opts[f"{ksp_c.getOptionsPrefix()}inner_ksp_type"] = "preonly"
            opts[f"{ksp_c.getOptionsPrefix()}inner_pc_type"] = "ilu"
            opts[f"{ksp_c.getOptionsPrefix()}inner_pc_factor_levels"] = 0
            opts[f"{ksp_c.getOptionsPrefix()}inner_pc_factor_fill"] = 2.0
            opts[f"{ksp_c.getOptionsPrefix()}upper_ksp_type"] = "preonly"
            opts[f"{ksp_c.getOptionsPrefix()}upper_pc_type"] = "ilu"
            opts[f"{ksp_c.getOptionsPrefix()}upper_pc_factor_levels"] = 0
            opts[f"{ksp_c.getOptionsPrefix()}upper_pc_factor_fill"] = 2.0

            for optk, optv in solver_params.AMG_TYPES[args.amg_type].items():
                opts[f"{ksp_c.getOptionsPrefix()}{optk}"] = optv

            ksp_u.setFromOptions()
            ksp_c.setFromOptions()
            snes.getKSP().setFromOptions()

            problem = solvers.NonlinearPDE_SNESProblem(F, J, sol_vars, bcs, P=P)
            snes.setFunction(problem.F_block, Fvec)
            snes.setJacobian(problem.J_block, J=Jmat, P=Pmat)
            snes.setFromOptions()
            snes.view()

            x = fem.petsc.create_vector_block(F)
            x.set(0.0)
            PETSc.Log().begin()
            t0 = time.time()
            snes.solve(None, x)
            t1 = time.time()
            PETSc.Sys.Print(f"SNES converged reason: {snes.getConvergedReason()}, solve time: {t1-t0:.3f}s")
            PETSc.Log().view(log_viewer)
            dt.value = 5 * dt.value
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
        # exact simulation time
        dt.value = 5 * dt.value #min(args.dt, TIME - t)
        # gamma.value = args.gamma * 10
        c0.x.array[:] = c.x.array
        cvtx.write(t)
        k = tdim - 2
        I_left = comm.allreduce(fem.assemble_scalar(fem.form(
                                inner(kappa_elec * phi_ref * L_ref ** (k) * grad(u_0), n) * ds(markers.left),
                                entity_maps=entity_maps)), op=MPI.SUM)
        I_right = comm.allreduce(fem.assemble_scalar(fem.form(
                                inner(kappa_pos_am * phi_ref * L_ref ** (k) * grad(u_1), n) * ds(markers.right),
                                entity_maps=entity_maps)), op=MPI.SUM)
        I_interface = comm.allreduce(fem.assemble_scalar(fem.form(
                                inner(faraday_const * D * c_ref * L_ref ** (k) * grad(c(r_res)), n_r) * dInterface,
                                entity_maps=entity_maps)), op=MPI.SUM)

        u_avg_right_tilde = comm.allreduce(fem.assemble_scalar(fem.form(u_1 * ds(markers.right),
                                                                        entity_maps=entity_maps)), op=MPI.SUM)
        u_avg_right = u_avg_right_tilde * phi_ref * L_ref ** 2 / A_right
        u_stdev_right_tilde = comm.allreduce(fem.assemble_scalar(fem.form(
                                            (u_1 - u_avg_right_tilde) ** 2 * ds(markers.right),
                                            entity_maps=entity_maps)), op=MPI.SUM)
        u_stdev_right = np.sqrt(u_stdev_right_tilde  * (phi_ref * L_ref ** 2) ** 2 / A_right)
        i_avg_se_am = I_interface / A_se_am
        i_avg_left = I_left / A_left
        i_avg_right = I_right / A_right

        i_stdev_left = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(
                                (kappa_elec * phi_ref * L_ref ** (-k) * inner(grad(u_0), n) - i_avg_left) ** 2 * ds(markers.left),
                                entity_maps=entity_maps)), op=MPI.SUM) / A_left_tilde)
        i_stdev_se_am = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(
                                (faraday_const * D * c_ref * L_ref ** (-k) * inner(grad(c(r_res)), n_r) - i_avg_se_am) ** 2 * dInterface,
                                entity_maps=entity_maps)), op=MPI.SUM) / A_se_am_tilde)
        i_stdev_right = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(
                                (kappa_pos_am * phi_ref * L_ref ** (-k) * inner(grad(u_1), n) - i_avg_right) ** 2 * ds(markers.right),
                                entity_maps=entity_maps)), op=MPI.SUM) / A_right_tilde)
        stats.append(
                     {
                     "t [s]": t * t_ref,
                    "I left [A]": I_left,
                    "I interface [A]": I_interface,
                    "I right [A]": I_right,
                    "I (target) right [A]": I_tot_,
                    "C-rate": args.C_rate,
                    "u (avg) right [V]": u_avg_right,
                    "u (stdev) right [v]": u_stdev_right,
                    "i (avg) left [A/m2]": i_avg_left,
                    "i (stdev) left [A/m2]": i_stdev_left,
                    "i (avg) se/am [A/m2]": i_avg_se_am,
                    "i (stdev) se/am [A/m2]": i_stdev_se_am,
                    "i (avg) right [A/m2]": i_avg_right,
                    "i (stdev) right [A/m2]": i_stdev_right,
                    "Diffusivity [m2/s]": args.D,
                    "Positive Wa": args.Wa_p,
                    "Kr": args.kr,
            }
                     )
    cvtx.close()
    with open(stats_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    time_elapsed = timeit.default_timer() - start_time

    metadata = {
        "I left [A]": I_left,
        "I interface [A]": I_interface,
        "I right [A]": I_right,
        "I (target) right [A]": I_tot_,
        "C-rate": args.C_rate,
        "u (avg) right [V]": u_avg_right,
        "u (stdev) right [v]": u_stdev_right,
        "i (avg) left [A/m2]": i_avg_left,
        "i (stdev) left [A/m2]": i_stdev_left,
        "i (avg) se/am [A/m2]": i_avg_se_am,
        "i (stdev) se/am [A/m2]": i_stdev_se_am,
        "i (avg) right [A/m2]": i_avg_right,
        "i (stdev) right [A/m2]": i_stdev_right,
        "time elapsed [s]": time_elapsed,
        "solve time [s]": t1 - t0,
        "L ref [m]": ref["L"],
        "R_p ref [m]": R_p_ref,
        "c ref [mol/m3]": ref["c"],
        "phi ref [V]": ref["phi"],
        "t ref [s]": ref["t"],
        "min time step [s]": args.dt * ref["t"],
        "Positive Wa": args.Wa_p,
        "Thiele modulus": thiele,
        "Diffusivity [m2/s]": args.D,
        "Kr": args.kr,
        "concentration field polynomial approximation order (p)": args.p_concentration,
        "potential field polynomial approximation order (p)": args.p_potential,
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
            c_json = {}
            p_json = {}
            all_c_vals = c_plot_vals
            all_u_vals = u_plot_vals
            for rank in range(1, comm_size):
                addtnl_c = comm.recv(source=rank, tag=11)
                all_c_vals = np.vstack((all_c_vals, addtnl_c))

                addtnl_u = comm.recv(source=rank, tag=13)
                all_u_vals = np.vstack((all_u_vals, addtnl_u))

            c_vals = all_c_vals[all_c_vals[:, 2].argsort()]
            u_vals = all_u_vals[all_u_vals[:, 2].argsort()]
            t_str = f"{t*t_ref:.3f}"

            c_json = {
                "t": t * t_ref,
                "Wa": args.Wa_p,
                "D": args.D,
                "kr": args.kr,
                "x": c_vals[:, 2].tolist(),
                "y": (c_vals[:, 3]*c_ref/c_max).tolist()
            }

            p_json = {
                "t": t * t_ref,
                "Wa": args.Wa_p,
                "D": args.D,
                "kr": args.kr,
                "x": u_vals[:, 2].tolist(),
                "y": u_vals[:, 3].tolist()
            }

            potential_json_path = os.path.join(results_dir, f"potential-{t_str}.json")
            concentration_json_path = os.path.join(results_dir, f"concentration-{t_str}.json")

            with open(potential_json_path, "w", encoding='utf-8') as f:
                json.dump(p_json, f, ensure_ascii=False, indent=4)

            with open(concentration_json_path, "w", encoding='utf-8') as f:
                json.dump(c_json, f, ensure_ascii=False, indent=4)

            fig, ax = plt.subplots()
            ax.plot(c_vals[:, 2], c_vals[:, 3]*c_ref/c_max, 'k', label=r'0.5$L_x$,0.5$L_y$', linewidth=1)
            ax.grid(True)
            ax.legend()
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_box_aspect(1)
            ax.set_ylabel(r'$\hat{c}$', rotation=90, labelpad=0, fontsize='xx-large')
            ax.set_xlabel(r'$\hat{x}$')
            ax.set_title(r'$\mathrm{Wa}$ = ' + f'{args.Wa_p}' + ',' + r'$\frac{\kappa}{\sigma}$ = ' + f'{args.kr}' + f' t = {t*t_ref:.3f}s')
            plt.tight_layout()
            plt.savefig(concentration_plot_file.replace(".eps", f"{t_str}.eps"))

            fig, ax = plt.subplots()
            ax.plot(u_vals[:, 2], u_vals[:, 3], 'k', label=r'0.5$L_x$,0.5$L_y$', linewidth=1)
            ax.grid(True)
            ax.legend()
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_box_aspect(1)
            ax.set_ylabel(r'$\hat{\phi}$', rotation=90, labelpad=0, fontsize='xx-large')
            ax.set_xlabel(r'$\hat{x}$')
            ax.set_title(r'$\mathrm{Wa}$ = ' + f'{args.Wa_p}' + ',' + r'$\frac{\kappa}{\sigma}$ = ' + f'{args.kr}' + f' t = {t*t_ref:.3f}s')
            plt.tight_layout()
            plt.savefig(potential_plot_file.replace(".eps", f"{t_str}.eps"))
