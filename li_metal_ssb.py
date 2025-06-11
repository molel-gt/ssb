# SPDX-License-Identifier: MIT
import argparse
import csv
import datetime
import json
import logging
import os
import resource
import sys
import time
import timeit

import basix
import dolfinx
import dolfinx.fem.petsc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import scifem
import scipy
import scipy.special as sp
import ufl
import warnings

# os.environ["XDG_CACHE_HOME"] = os.path.join(os.getcwd(), ".cache/fenics", str(hash(tuple(sys.argv))))
from mpi4py import MPI

from dolfinx import cpp, default_real_type, default_scalar_type, fem, io, jit, mesh, log
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.graph import partitioner_scotch
from dolfinx.nls import petsc as petsc_nls
from matplotlib import rc
from packaging.version import Version
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
hold_voltage = "hold_voltage"
cyclic_voltammetry = "cyclic_voltammetry"
gitt = "gitt_titration"
rest = "rest"
I_rest = np.finfo(np.float64).eps
max_rest_c_rate = 0.001 # maximum c-rate to be considered rest

modes = (galvanostatic, potentiostatic)
micron = 1e-6
V_MAX = 5.0  # upper cutoff voltage
V_MIN = 2.5  # lower cutoff voltage
c_max = 35000
directions = {'x': 0, 'y': 1, 'z': 2}
MIN_C_RATE = 0.01
EPS = 1e-14
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


def eta_s(kappa, u, n, i0, kinetics_type='linear', ref={"L": 1, "phi": 1, "t": 1, "c": 1}):
    if isinstance(kappa, list):
        i_loc = -0.5 * ref["phi"] / ref["L"] * (kappa[0] * inner(grad(u[0]), n[1]) + kappa[1] * inner(grad(u[1]), n[1]))
    else:
        i_loc = -inner((kappa * grad(u)), n) * ref["phi"] / ref["L"]
    if kinetics_type == "butler_volmer":
        return 2 * ufl.ln(0.5 * i_loc/i0 + ufl.sqrt((0.5 * i_loc/i0)**2 + 1)) * (R * T / (faraday_const * ref["phi"]))
    elif kinetics_type == "linear":
        return R * T * i_loc / (i0 * faraday_const * ref["phi"])
    elif kinetics_type == "tafel":
        return ufl.sign(i_loc) * R * T / (0.5 * faraday_const * ref["phi"]) * ufl.ln(np.abs(i_loc)/i0)


def arctanh(y):
    return 0.5 * ufl.ln((1 + y) / (1 - y))


def U_ocp(c, cmax=1.0, phi_ref=V_MAX):
    """
    Chen2020 OCP for NMC622 + bound-checking
    """
    return  1/phi_ref * (4.4875 - 0.8090 * c/cmax - 0.0428 * ufl.tanh(18.5138*(c/cmax - 0.5542)) +\
        -17.7326 * ufl.tanh(15.7890*(c/cmax - 0.3117)) + 17.5842 * ufl.tanh(15.9308*(c/cmax - 0.3120))) * ufl.conditional(ufl.ge(c/c_max, 0), 1, 0) * ufl.conditional(ufl.le(c/c_max, 1), 1, 0)+\
        1/phi_ref * (
         ufl.conditional(ufl.lt(c/c_max, 0), 4.6785099 - 1000 * c/c_max, 0) +\
         ufl.conditional(ufl.gt(c/c_max, 1.0), 3.4873 - 1000 * c/c_max, 0)
         )


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


def IS_chainsum(IS_main, parts):
    if len(parts) == 0:
        return IS_main

    if len(parts) == 1:
        return IS_main.sum(parts[0])

    return IS_chainsum(IS_main.sum(parts[0]), parts[1:])


class CCCV_Cycler:
    def __init__(self, dt, ref, input_json):
        self._modes_input_json = input_json
        self._n_remaining = len(modes)
        self._modes = None
        self._ref = ref
        self._mode_idx = 0
        self._time = 0
        self._dt = dt
        self._current_mode_time = 0
        self._sod = 0
        self._stop = False
        self._cv_data = None
        self._gitt_data = None

    @property
    def ref(self):
        return self._ref

    @property
    def sod(self):
        return self._sod

    @property
    def time(self):
        return self._time

    @property
    def modes(self):
        return self._modes

    @property
    def mode_idx(self):
        return self._mode_idx

    @property
    def t_max(self):
        return np.sum([m["time"] for m in self.modes])

    @property
    def current_mode(self):
        return self.modes[self.mode_idx]

    @property
    def current_mode_time(self):
        return self._current_mode_time

    @property
    def current_mode_type(self):
        if self.cv_data is not None:
            return cyclic_voltammetry

        if self.gitt_data is not None:
            return gitt

        if self.current_mode["c-rate"] is not None:
            return galvanostatic

        if self.current_mode["voltage"] is not None:
            return potentiostatic

    @property
    def dt(self):

        res = self.current_mode["time"] - self.current_mode_time
        if np.isclose(res, 0):
            return self._dt

        return np.min([self._dt, res])

    @property
    def stop(self):
        return self._stop

    @property
    def cv_data(self):
        return self._cv_data

    @property
    def gitt_data(self):
        return self._gitt_data

    def gitt_time(self, t0):
        return (self.time - t0) % self.gitt_data["period"]

    def gitt_current_function(self, t0=0):
        return (0 <= self.gitt_time(t0) <= self.gitt_data["pulse-time"]) * self.gitt_data["c-rate"]  +\
         (self.gitt_data["pulse-time"] < self.gitt_time(t0) <= self.gitt_data["period"]) * MIN_C_RATE

    def cv_voltage_function(self, t0=0):
        _scan_rate = self.cv_data["scan-rate"]
        _deltaV = self.cv_data["deltaV"]
        _V = self.cv_data["V"]
        _scan_time = (_deltaV / _scan_rate) / self.ref["t"]
        _scan_rate *= ref["t"]
        return (t0 <= self.time <= t0 + _scan_time) * (_V + _scan_rate * (self.time - t0))+\
            (t0 + _scan_time < self.time <= t0 + 2 * _scan_time) * (_V + _deltaV - _scan_rate * (self.time - t0 - _scan_time))+\
            (t0 + 2 * _scan_time < self.time <= t0 + 3 * _scan_time) * (_V - _scan_rate * (self.time - t0 - 2 * _scan_time))+\
            (t0 + 3 * _scan_time < self.time <= t0 + 4 * _scan_time) * (_V - _deltaV + _scan_rate * (self.time - t0 - 3 * _scan_time))

    def setup(self):
        if self.modes is None:
            with open(self._modes_input_json) as fp:
                data = json.load(fp)
                cccv_data = data.get("data")
                self._cv_data = data.get("cv")
                self._gitt_data = data.get("gitt")
                self._sod = data["sod"]

                if self.cv_data is not None:
                    _scan_rate = self.cv_data["scan-rate"]
                    _deltaV = self.cv_data["deltaV"]
                    _V = self.cv_data["V"]
                    _scan_time = (_deltaV / _scan_rate) / self.ref["t"]
                    self._modes = [{"time": 4 * _scan_time, "c-rate": None, "direction": np.nan}]

                if self.gitt_data is not None:
                    _c_rate = self.gitt_data["c-rate"]
                    _n_cycles = self.gitt_data["n-cycles"]
                    _pulse_time = self.gitt_data["pulse-time"]
                    _rest_time = self.gitt_data["rest-time"]
                    _V_min = self.gitt_data["stop"]["V_min"]
                    _V_max = self.gitt_data["stop"]["V_max"]

                    self._gitt_data["pulse-time"] = _pulse_time / self.ref["t"]
                    self._gitt_data["rest-time"] = _rest_time / self.ref["t"]
                    self._gitt_data["period"] = _pulse_time/ self.ref["t"] + _rest_time/ self.ref["t"]
                    self._modes = [{"time": _n_cycles * (self._gitt_data["period"]), "c-rate": _c_rate, "direction": 1}]

                if cccv_data is not None:
                    for idx, _row in enumerate(cccv_data):
                        _row["time"] = _row["time"] / self.ref["t"]
                        if _row['c-rate'] is None:
                            continue
                        if _row["c-rate"] > max_rest_c_rate and np.isclose(_row['direction'], 0):
                            raise ValueError("c-rate is greater than maximum for rest phase")
                    self._modes = cccv_data

                    self._time += self.dt
                    self._current_mode_time += self.dt

    def next(self):
        self._time += self.dt
        self._current_mode_time += self.dt

        if self.current_mode_time > self.current_mode["time"] and self.mode_idx < len(self.modes) - 1:
            self._mode_idx += 1
            self._current_mode_time = self.dt
            return True
        return False

    def check_stop_criteria(self, V_cell, I_cell):
        if self.current_mode_type == cyclic_voltammetry and self.time >= self.t_max:
            self._stop = True
            return

        if self.current_mode_type == gitt and self.time >= self.t_max:
            self._stop = True
            return

        if self.mode_idx >= len(self.modes):
            self._stop = True
            return

        # if self.time >= self.t_max:
        #     self._stop = True
        #     return

        # stop at global upper and lower cutoff voltage
        if V_cell >= V_MAX or V_cell <= V_MIN:
            self._stop = True
            return

        # constant current mode: stop at maximum voltage during charge, minimum voltage during discharge
        if self.current_mode_type == galvanostatic:
            if self.current_mode["direction"] < 0 and V_cell >= self.current_mode['stop']['V_max']:
                self._stop = True
                return

            if self.current_mode["direction"] > 0 and V_cell <= self.current_mode['stop']['V_min']:
                self._stop = True
                return

        # constant voltage mode: stop at maximum current during charge, minimum current during discharge
        if self.current_mode_type == potentiostatic:
            if self.current_mode["direction"] > 0 and I_cell >= self.current_mode["stop"]["I_max"]:
                self._stop = True
                return

            if self.current_mode["direction"] < 0 and I_cell <= self.current_mode["stop"]["I_min"]:
                self._stop = True
                return

        return


def frequency_condition(values, vleft, vright, tol_fun_left, tol_fun_right):
    tol_fun_left.interpolate(lambda x: vleft * (x[0] + EPS) / (x[0] + EPS))
    tol_fun_right.interpolate(lambda x: vright * (x[0] + EPS) / (x[0] + EPS))
    return ufl.conditional(ufl.ge(values, tol_fun_left), 1, 0) * ufl.conditional(ufl.lt(values, tol_fun_right), 1, 0)


def current_density_distribution(comm, current_h, n, tol_fun_left, tol_fun_right, ds_, entity_maps, intervals):
    """
    For 0 <= k < N, compute area of interface with current density i(k) <= i < i(k+1).
    """
    densities = np.zeros((len(intervals)-1, 3))
    for idx in range(len(intervals) - 1):
        i_min = intervals[idx]
        i_max = intervals[idx+1]
        i_density = comm.allreduce(fem.assemble_scalar(fem.form(frequency_condition(np.abs(ufl.inner(current_h, n)), i_min, i_max, tol_fun_left, tol_fun_right) * ds_, entity_maps=entity_maps)), op=MPI.SUM)
        densities[idx, 0] = i_min
        densities[idx, 1] = i_max
        densities[idx, 2] = i_density

    return densities


def communicate_indices(
    index_map: dolfinx.common.IndexMap, local_indices: npt.NDArray[np.int32]
) -> npt.NDArray[np.int32]:
    """
    Given a set of local indices find ghosted indices marked by any
    other other process (other ghosted processes or owner process).
    Args:
        index_map: Map describing the ownership structure of the
            entities
        local_indices: List of indices local to process that should
            be distributed
    Returns:
        All indices (local index) on process (including ghosts) that
        have been marked by this or and other process.
    """
    index_accumulator = dolfinx.la.vector(index_map, 1)
    index_accumulator.array[:] = 0
    index_accumulator.array[local_indices] = 1
    index_accumulator.scatter_reverse(dolfinx.la.InsertMode.add)
    return np.flatnonzero(index_accumulator.array).astype(np.int32)

def compute_interface_data(
    cell_tags: dolfinx.mesh.MeshTags, facet_indices: npt.NDArray[np.int32]
) -> npt.NDArray[np.int32]:
    """
    Compute interior facet integrals that are consistently ordered according to the `cell_tags`,
    such that the data `(cell0, facet_idx0, cell1, facet_idx1)` is ordered such that
    `cell_tags[cell0]`>`cell_tags[cell1]`, i.e the cell with the highest cell marker is considered the
    "+" restriction".

    Args:
        cell_tags: MeshTags that must contain an integer marker for all cells adjacent to the `facet_indices`
        facet_indices: List of facets (local index) that are on the interface.
    Returns:
        The integration data.
    """
    # Future compatibilty check
    integration_args: tuple[int] | tuple
    if Version("0.10.0") <= Version(dolfinx.__version__):
        integration_args = ()
    else:
        fdim = cell_tags.dim - 1
        integration_args = (fdim,)
    idata = cpp.fem.compute_integration_domains(
        dolfinx.fem.IntegralType.interior_facet,
        cell_tags.topology,
        facet_indices,
        *integration_args,
    )
    ordered_idata = idata.reshape(-1, 4).copy()
    switch = cell_tags.values[ordered_idata[:, 0]] < cell_tags.values[ordered_idata[:, 2]]
    if True in switch:
        ordered_idata[switch, :] = ordered_idata[switch][:, [2, 3, 0, 1]]
    return ordered_idata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='secondary current distribution')
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1.0, type=float)
    parser.add_argument("--Wa_n", help="Wagna number for negative electrode: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e-3, type=float)
    parser.add_argument("--Wa_p", help="Wagna number for positive electrode: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e3, type=float)
    parser.add_argument("--D", help="Diffusivity [m2/s]", nargs='?', const=1, default=1e-14, type=float)
    parser.add_argument("--kr", help="ratio of ionic to electronic conductivity", nargs='?', const=1, default=1, type=float)
    parser.add_argument("--gamma", help="interior penalty parameter", nargs='?', const=1, default=1e-2, type=float)
    parser.add_argument("--alpha", help="interior penalty parameter", nargs='?', const=1, default=0, type=float)
    parser.add_argument("-p_c", "--p_concentration", help="polynomial approximation order for concentration field", nargs='?', const=1, default=4, type=int)
    parser.add_argument("-p_u0", "--p_u0", help="polynomial approximation order for potential field in SE", nargs='?', const=1, default=2, type=int)
    parser.add_argument("-p_u1", "--p_u1", help="polynomial approximation order for potential field in AM", nargs='?', const=1, default=1, type=int)
    parser.add_argument("-cell_type", "--cell_type", help="cell type to use", nargs='?', const=1, default="tetrahedron", type=str)
    parser.add_argument("-dt", "--dt", help="minimum normalized time step", nargs='?', const=1, default=2e-7, type=float)
    parser.add_argument('--cycle_name', help='cycle name for identification', nargs='?', const=1, default='charge', type=str)
    parser.add_argument("-cycling_json", "--cycling_json", help="cycling input data", nargs='?', const=1, default="cycling.json", type=str)
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
    parser.add_argument("--improved_guess", help="whether to solve for improved guess", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--compute_distribution", help="compute current distribution stats", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--nested_fieldsplit", help="whether to use chain of fieldsplit preconditioners", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    # PETSc.Sys.Print("************************************** CYCLING PARAMETERS SUMMARY *************************************")
    PETSc.Sys.Print(utils.starpad(" CYCLING PARAMETERS SUMMARY "))
    PETSc.Sys.Print("Positive electrode Wa                                  :", args.Wa_p)
    PETSc.Sys.Print("Conductivity ratio (Kr)                                :", args.kr)
    PETSc.Sys.Print("Lithium diffusivity in positive active material [m2/s] :", args.D)
    PETSc.Sys.Print("Kinetics                                               :", args.kinetics)
    PETSc.Sys.Print(utils.starpad("*"))
    PETSc.Sys.Print(" SOLVER PARAMETERS ".center(90, "*"))
    PETSc.Sys.Print("interior penalty parameter (gamma)                     :", args.gamma)
    PETSc.Sys.Print("solve improved guesss                                  :", args.improved_guess)
    PETSc.Sys.Print("minimum dt [s]                                         :", args.dt)
    PETSc.Sys.Print(utils.starpad("*"))

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
    phi_ref = V_MAX
    c_ref = c_max
    # c_ref = kappa_pos_am * phi_ref / (faraday_const * D)
    ref = {"t": t_ref, "phi": phi_ref, "c": c_ref, "L": L_ref}
    args.dt = args.dt / ref["t"] # scale input

    cycler = CCCV_Cycler(ref=ref, dt=args.dt, input_json=args.cycling_json)
    cycler.setup()

    # exchange current densities
    i0_n = kappa_elec * R * T / (Wa_n * faraday_const * L_ref)
    i0_p = kappa_elec * R * T / (Wa_p * faraday_const * L_ref)
    R_p_ref = 10e-6  # characteristic diffusion length

    thiele = R_p_ref * i0_p * V_MAX / (R * T * D * c_max)

    c_init = cycler.sod

    output_meshfile = os.path.join(args.mesh_folder, "mesh.msh")
    results_dir = os.path.join(args.mesh_folder, args.kinetics,
                               str(Wa_n) + "-" + str(Wa_p) + "-" + str(args.kr),
                               f'{args.p_u0}-{args.p_u1}-{args.p_concentration}', args.cycle_name,
                               str(args.gamma) + "-" + str(args.alpha), str(comm.Get_size())
                               )
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
    interface_i_density_file = os.path.join(results_dir, "i_x_density.json")
    stats_csv_file = os.path.join(results_dir, "stats.csv")
    i_interface_density_plot = os.path.join(results_dir, "i_x_density.eps")
    convergence_history = os.path.join(results_dir, "convergence.eps")
    se_am_frequency_plot = os.path.join(results_dir, "se_am_frequency.eps")
    se_am_cdf_plot = os.path.join(results_dir, "se_am_cdf.eps")
    log_datafile = os.path.join(results_dir, "log.txt")

    # load mesh
    partitioner = mesh.create_cell_partitioner(partitioner_scotch(), mesh.GhostMode.shared_facet)
    domain, ct, ft = io.gmshio.read_from_msh(output_meshfile, comm, partitioner=partitioner)[:3]
    tdim = domain.topology.dim
    fdim = tdim - 1
    k = tdim - 2
    domain.topology.create_connectivity(tdim, fdim)
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(fdim, fdim)
    ct_imap = domain.topology.index_map(tdim)
    num_entities_local = ct_imap.size_local + ct_imap.num_ghosts
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
    parent_to_sub_electrolyte = np.full(num_entities_local, -1, dtype=np.int32)
    parent_to_sub_electrolyte[submesh_electrolyte_to_mesh] = np.arange(len(submesh_electrolyte_to_mesh), dtype=np.int32)
    parent_to_sub_positive_am = np.full(num_entities_local, -1, dtype=np.int32)
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

    ##
    Q = fem.functionspace(domain, ("DG", 0))
    kappa = fem.Function(Q, name='conductivity')
    kappa_total = kappa_elec + kappa_pos_am

    cells_elec = ct.find(markers.electrolyte)
    kappa.x.array[cells_elec] = np.full_like(cells_elec, kappa_elec / kappa_total, dtype=default_scalar_type)

    # kappa_pos_am = kappa_elec/args.kr
    cells_pos_am = ct.find(markers.positive_am)
    kappa.x.array[cells_pos_am] = np.full_like(cells_pos_am, kappa_pos_am / kappa_total, dtype=default_scalar_type)
    ##

    u_0, F_00, m_to_elec = define_interior_eq(domain, args.p_u0, submesh_electrolyte, submesh_electrolyte_to_mesh, 0.0, kappa, cell_type)
    u_1, F_11, m_to_pos_am = define_interior_eq(domain, args.p_u1, submesh_positive_am, submesh_positive_am_to_mesh, 0.0, kappa, cell_type)
    u_0.name = "u_b"
    u_1.name = "u_t"

    # initial guess
    # u_0.interpolate(lambda x: x[0]-x[0])
    # u_1.interpolate(lambda x: 0.5 + x[0]-x[0])

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

    # ordered_integration_data = compute_interface_data(ct, charge_xfer_facets)
    # # Pad entity maps for sparsity pattern
    # parent_cells_plus = ordered_integration_data[:, 0]
    # parent_cells_minus = ordered_integration_data[:, 2]
    # entity_maps[submesh_electrolyte][parent_cells_minus] = entity_maps[submesh_electrolyte][parent_cells_plus]
    # entity_maps[submesh_positive_am][parent_cells_plus] = entity_maps[submesh_positive_am][parent_cells_minus]
    # ordered_integration_data = ordered_integration_data.flatten()
    # integral_data_interface = [(markers.electrolyte_v_positive_am, ordered_integration_data)]
    # dInterface = ufl.Measure(
    #     "dS",
    #     domain=domain,
    #     subdomain_data=integral_data_interface,
    #     subdomain_id=markers.electrolyte_v_positive_am)
    dx = ufl.Measure('dx', domain=domain, subdomain_data=ct)
    dx_r = ufl.Measure('dx', domain=domain, subdomain_data=ct, subdomain_id=markers.positive_am)
    dx_c = ufl.Measure('dx', domain=submesh_positive_am)
    ds = ufl.Measure('ds', domain=domain, subdomain_data=ft)
    ds_c = ufl.Measure('ds', domain=submesh_positive_am, subdomain_data=ft_positive_am)

    vol_pos_am_tilde = comm.allreduce(fem.assemble_scalar(fem.form(1 * dx(markers.positive_am), entity_maps=entity_maps)), op=MPI.SUM)
    vol_pos_am = vol_pos_am_tilde * L_ref ** tdim
    _c_rate = 0.01
    if cycler.current_mode["c-rate"] is not None:
        _c_rate = cycler.current_mode["c-rate"] * cycler.current_mode["direction"]
    I_tot_ = utils.get_c_rate_current(c_max, _c_rate, vol_pos_am)
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
    h = ufl.CellDiameter(domain)
    h_1 = ufl.CellDiameter(submesh_positive_am)
    h_l = h(l_res)
    h_r = h(r_res)

    # concentration problem
    dt = fem.Constant(submesh_positive_am, args.dt)
    el = basix.ufl.element(basix.ElementFamily.P, cell_type, args.p_concentration, basix.LagrangeVariant.gll_isaac, dtype=dolfinx.default_real_type)
    VC = fem.functionspace(submesh_positive_am, el)

    c, q = fem.Function(VC), ufl.TestFunction(VC)
    c0 = fem.Function(VC)
    u_int = fem.Function(VC)

    c0.interpolate(lambda x: x[0] - x[0] + c_init)
    c.interpolate(lambda x: x[0] - x[0] + c_init)

    q_r = ufl.TestFunction(c.function_space)(r_res)
    q_l = ufl.TestFunction(c.function_space)(l_res)
    c_r = c(r_res)

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
    A_left = A_left_tilde * (L_ref ** (k+1))

    A_right_tilde = comm.allreduce(fem.assemble_scalar(fem.form(1 * ds(markers.right))), op=MPI.SUM)
    A_right = A_right_tilde * (L_ref ** (k+1))

    A_se_am_tilde = comm.allreduce(fem.assemble_scalar(fem.form(1 * ds_c(markers.electrolyte_v_positive_am))), op=MPI.SUM)
    A_se_am = A_se_am_tilde * (L_ref ** (k+1))
    A_se_am_to_vol_am = A_se_am / vol_pos_am
    PETSc.Sys.Print("Area Left [m2]                        :", f"{A_left:.1e}")
    PETSc.Sys.Print("Area Right [m2]                       :", f"{A_right:.1e}")
    PETSc.Sys.Print("Area SE/AM [m2]                       :", f"{A_se_am:.1e}")
    PETSc.Sys.Print("SE/AM area to cross-section area      :", f"{A_se_am/A_right:,.0f}")
    PETSc.Sys.Print("SE/AM area to volume ratio            :", f"{A_se_am_to_vol_am:,.0f}")

    R_right = scifem.create_real_functionspace(submesh_facets_right)
    if args.cell_type == "tetrahedron":
        _2d_shape = basix.CellType.triangle
    elif args.cell_type == "hexahedron":
        _2d_shape = basix.CellType.quadrilateral
    elif args.cell_type == "triangle":
        _2d_shape = basix.CellType.interval
    else:
        raise ValueError("Unknown cell type")
    el_V_r = basix.ufl.element(basix.ElementFamily.P, _2d_shape, args.p_u1, basix.LagrangeVariant.gll_isaac, dtype=dolfinx.default_real_type)
    V_r = fem.functionspace(submesh_facets_right, el_V_r)
    lmbda, mu = fem.Function(V_r), ufl.TestFunction(V_r)

    V_cell, w = fem.Function(R_right), ufl.TestFunction(R_right)

    I_tot = fem.Constant(submesh_facets_right, PETSc.ScalarType(I_tot_))
    I_tot_tilde = fem.Constant(submesh_facets_right, PETSc.ScalarType(I_tot.value /(L_ref ** (k) * kappa_total * phi_ref)))
    gamma = fem.Constant(domain, PETSc.ScalarType(args.gamma))
    h_avg = 0.5 * (h_l + h_r)
    kappa_l = kappa(l_res)
    kappa_r = kappa(r_res)

    F_0 = (
        - 0.5 * mixed_term(kappa_l * u_l + kappa_r * u_r, v_l, n_l) * dInterface
        - 0.5 * mixed_term(kappa_l * v_l, (u_r - u_l - eta_s(kappa_pos_am, u_r, n_r, i0_p, kinetics_type=args.kinetics, ref=ref) - U_ocp(c_r)), n_l) * dInterface
    )

    F_1 = (
        + 0.5 * mixed_term(kappa_l * u_l + kappa_r * u_r, v_r, n_l) * dInterface
        - 0.5 * mixed_term(kappa_r * v_r, (u_r - u_l - eta_s(kappa_pos_am, u_r, n_r, i0_p, kinetics_type=args.kinetics, ref=ref) - U_ocp(c_r)), n_l) * dInterface
    )
    F_0 += - gamma / h_avg * (u_r - u_l - eta_s(kappa_pos_am, u_r, n_r, i0_p, kinetics_type=args.kinetics, ref=ref) - U_ocp(c_r)) * v_l * dInterface
    F_1 += + gamma / h_avg * (u_r - u_l - eta_s(kappa_pos_am, u_r, n_r, i0_p, kinetics_type=args.kinetics, ref=ref) - U_ocp(c_r)) * v_r * dInterface

    F_0 += F_00
    F_1 += F_11

    # # additional penalty terms, e.g. 5e3, or (1 + Wa)*(1 + Kr)/(1 + Wa * Kr)
    if args.alpha > 0:
        F_0 += - kappa_l * args.alpha * h_avg * inner(-kappa_l * grad(u_l) + kappa_r * grad(u_r), grad(v_l)) * dInterface
        F_1 += + kappa_r * args.alpha * h_avg * inner(-kappa_l * grad(u_l) + kappa_r * grad(u_r), grad(v_r)) * dInterface

    F_1_cv = 0
    F_1_cv += F_1
    F_1_cc = 0
    F_1_cc += F_1
    F_1_cc += - v_1 * lmbda * ds_f(3)
    F_1a = (V_cell - u_1) * mu * ds_f(3)
    F_1b = w * (I_tot_tilde / A_right_tilde + lmbda) * ds_f(3)

    F_2 = (c - c0)/dt * q * dx_r + inner(ufl.grad(c), ufl.grad(q)) * dx_r
    F_2 += -inner(kappa_total * phi_ref/(D * faraday_const * c_ref)/2 * (kappa_l * grad(u_l) + kappa_r * grad(u_r)), n_r) * q_r * dInterface
    # F_2 += -(L_ref/(faraday_const * D * c_ref)) * 2*i0_p * (ufl.sinh(0.5 * phi_ref * (u_r - u_l - U_ocp(c_r)) * faraday_const / (R * T))) * q_r * dInterface
    F_2 += -1e4 * gamma * h_r * inner(kappa_total * phi_ref/(D * faraday_const * c_ref)/2 * grad(kappa_l * u_l + kappa_r * u_r) - grad(c_r), n_r) * inner(grad(q_r), n_r) * dInterface

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
    if cycler.current_mode_type in (potentiostatic, cyclic_voltammetry):
        bcs = [bc_left, bc_right]

    J_cc = None
    J_cv = None

    for mode in [cyclic_voltammetry, gitt, galvanostatic, potentiostatic]:
        if mode in (galvanostatic, gitt):
            jac00 = ufl.derivative(F_0, u_0)
            jac01 = ufl.derivative(F_0, u_1)
            jac02 = ufl.derivative(F_0, lmbda)
            jac03 = ufl.derivative(F_0, V_cell)
            jac04 = ufl.derivative(F_0, c)

            jac10 = ufl.derivative(F_1_cc, u_0)
            jac11 = ufl.derivative(F_1_cc, u_1)
            jac12 = ufl.derivative(F_1_cc, lmbda)
            jac13 = ufl.derivative(F_1_cc, V_cell)
            jac14 = ufl.derivative(F_1_cc, c)

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

            J_cc = [
                [J00, J01, J02, J03, J04],
                [J10, J11, J12, J13, J14],
                [J20, J21, J22, J23, J24],
                [J30, J31, J32, J33, J34],
                [J40, J41, J42, J43, J44],
            ]

        elif mode in (potentiostatic, cyclic_voltammetry):
            jac00 = ufl.derivative(F_0, u_0)
            jac01 = ufl.derivative(F_0, u_1)
            jac02 = ufl.derivative(F_0, c)

            jac10 = ufl.derivative(F_1_cv, u_0)
            jac11 = ufl.derivative(F_1_cv, u_1)
            jac12 = ufl.derivative(F_1_cv, c)

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

            J_cv = [
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

    ############################################################################
    # constant current:
    F_cc = [
        fem.form(F_0, entity_maps=entity_maps),
        fem.form(F_1_cc, entity_maps=entity_maps),
        fem.form(F_1a, entity_maps=entity_maps),
        fem.form(F_1b, entity_maps=entity_maps),
        fem.form(F_2, entity_maps=entity_maps),
    ]

    n_dofs_cc = V0_map.size_global*V0.dofmap.index_map_bs + V1_map.size_global*V1.dofmap.index_map_bs + VC_map.size_global*VC.dofmap.index_map_bs +\
            V_r_map.size_global*V_r.dofmap.index_map_bs + R_right_map.size_global*R_right.dofmap.index_map_bs
    # constant voltage
    F_cv = [
        fem.form(F_0, entity_maps=entity_maps),
        fem.form(F_1_cv, entity_maps=entity_maps),
        fem.form(F_2, entity_maps=entity_maps),
    ]
    n_dofs_cv = V0_map.size_global*V0.dofmap.index_map_bs + V1_map.size_global*V1.dofmap.index_map_bs + VC_map.size_global*VC.dofmap.index_map_bs

    
    n_dofs = n_dofs_cc
    PETSc.Sys.Print(f"Setting up problem Wa: {args.Wa_p}, Kr: {args.kr}, #DoFs: {n_dofs:,}, nprocs: {comm.Get_size()}")

    log_viewer = PETSc.Viewer().STDOUT()
    log_viewer.setFileName(log_datafile)
    petsc_options = PETSc.Options()
    # stats = []
    sample_row = {
                    "t [s]": np.nan,
                    "I left [A]": np.nan,
                    "I interface [A]": np.nan,
                    "I interface (potential left) [A]": np.nan,
                    "I interface (potential right) [A]": np.nan,
                    "I (butler-volmer) [A]": np.nan,
                    "I right [A]": np.nan,
                    "I (target) right [A]": np.nan,
                    "u (avg) left [V]": np.nan,
                    "u (stdev) left [v]": np.nan,
                    "u (avg) right [V]": np.nan,
                    "u (stdev) right [v]": np.nan,
                    "surface overpotential (avg) [V]": np.nan,
                    "V (ocp) (avg) [V]": np.nan,
                    "i (avg) left [A/m2]": np.nan,
                    "i (stdev) left [A/m2]": np.nan,
                    "i (avg) se/am [A/m2]": np.nan,
                    "i (stdev) se/am [A/m2]": np.nan,
                    "i (avg) right [A/m2]": np.nan,
                    "i (stdev) right [A/m2]": np.nan,
                    "c surf (avg) (normalized)": np.nan,
                    "c surf (stdev) (normalized)": np.nan,
                    "c (avg) (normalized)": np.nan,
                    "I_interface error norm (normalized)": np.nan,
                    "I_interface error norm c (normalized)": np.nan,
                    "Diffusivity [m2/s]": np.nan,
                    "Positive Wa": np.nan,
                    "Kr": np.nan,
            }
    fp = open(stats_metadata_file, "w")
    stats_writer = csv.DictWriter(fp, fieldnames=sample_row.keys())
    stats_writer.writeheader()
    fp.flush()
    ########################################################################################################################################
    ## solve initial potential distribution at t = 0
    if args.improved_guess:
        PETSc.Sys.Print("************Begin Solve for t = 0 Potential Distribution*******************")
        if cycler.current_mode_type in (galvanostatic, gitt):
            n_dofs_t0 = V0_map.size_global*V0.dofmap.index_map_bs + V1_map.size_global*V1.dofmap.index_map_bs +\
                V_r_map.size_global*V_r.dofmap.index_map_bs + R_right_map.size_global*R_right.dofmap.index_map_bs
            F2D = F_cc[:4]
            J2D = [j2d[:4] for j2d in J_cc[:4]]
        elif cycler.current_mode_type in (potentiostatic, cyclic_voltammetry):
            n_dofs_t0 = V0_map.size_global*V0.dofmap.index_map_bs + V1_map.size_global*V1.dofmap.index_map_bs
            F2D = F_cv[:2]
            J2D = [j2d[:2] for j2d in J_cc[:2]]
        petsc_options.clear()
        J2D = fem.form(J2D)
        F2D = fem.form(F2D)
        Jmat2d = fem.petsc.create_matrix(J2D)
        Fvec2d = fem.petsc.create_vector(F2D, kind="mpi")
        snes = PETSc.SNES().create(comm)
        snes.setType('newtonls')
        snes.setTolerances(rtol=1e-7, max_it=200)
        snes.setMonitor(lambda _, it, residual: PETSc.Sys.Print("it:", it, "res:", residual))

        # set preconditioners
        petsc_options['log_view'] = None
        if args.nested_fieldsplit:
            J2d_mat = fem.petsc.create_matrix(J2D, kind="nest")
            nested_IS = J2d_mat.getNestISs()
            IS_u0 = nested_IS[0][0]
            IS_u1 = nested_IS[0][1]
            IS_l = nested_IS[0][2]
            IS_v = nested_IS[0][3]
            IS_u = IS_u0.sum(IS_u1)
            IS_lv = IS_l.sum(IS_v)
            IS_u1lv = IS_u1.sum(IS_lv)
            IS_ur = IS_chainsum(IS_u1, [IS_l, IS_v])

            snes.getKSP().setType(PETSc.KSP.Type.FGMRES)
            snes.getKSP().getPC().setType("fieldsplit")
            snes.getKSP().getPC().setFieldSplitIS(("ul", IS_u0), ("ur", IS_ur))
            petsc_options = PETSc.Options()
            petsc_options[f'{snes.getKSP().getOptionsPrefix()}ksp_gmres_restart'] = 100
            for kopt, vopt in solver_params.LINESEARCH.items():
                petsc_options[kopt] = vopt

            # petsc_options[f"{snes.getKSP().getOptionsPrefix()}pc_fieldsplit_off_diag_use_amat"] = True
            petsc_options[f"{snes.getKSP().getOptionsPrefix()}pc_fieldsplit_detect_saddle_point"] = True

            ksp_ul, ksp_ur = snes.getKSP().getPC().getFieldSplitSubKSP()
            # snes.getKSP().getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
            snes.getKSP().getPC().setFieldSplitSchurPreType(PETSc.PC.SchurPreType.SELFP)
            snes.getKSP().getPC().setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)

            ksp_ul.setType(PETSc.KSP.Type.CG)
            ksp_ul.getPC().setType(PETSc.PC.Type.ILU)
            ksp_ul.setTolerances(rtol=1e-7, max_it=1000)
            petsc_options[f"{ksp_ul.getOptionsPrefix()}pc_factor_levels"] = 0
            petsc_options[f"{ksp_ul.getOptionsPrefix()}pc_factor_fill"] = 2.0

            ksp_ur.setType(PETSc.KSP.Type.PREONLY)
            ksp_ur.getPC().setType(PETSc.PC.Type.ILU)
            ksp_ur.setTolerances(rtol=1e-7, max_it=1000)
            petsc_options[f"{ksp_ur.getOptionsPrefix()}pc_factor_levels"] = 0
            petsc_options[f"{ksp_ur.getOptionsPrefix()}pc_factor_fill"] = 2.0
        else:
            snes.getKSP().setType(PETSc.KSP.Type.FGMRES)
            snes.getKSP().getPC().setType(PETSc.PC.Type.ILU)
            snes.getKSP().setOptionsPrefix("snes_")
            snes.getKSP().setOperators(Jmat2d, Jmat2d)
            snes.getKSP().setTolerances(rtol=1e-7)
            snes.setErrorIfNotConverged(True)
            snes.getKSP().setErrorIfNotConverged(True)
            snes.getKSP().setConvergenceHistory()
            for kopt, vopt in solver_params.LINESEARCH.items():
                    petsc_options[kopt] = vopt
            petsc_options[f"{snes.getKSP().getOptionsPrefix()}pc_factor_levels"] = 0
            petsc_options[f"{snes.getKSP().getOptionsPrefix()}pc_factor_fill"] = 2.0
            snes.getKSP().setFromOptions()
            snes.setFromOptions()
            snes.view()
        soln_vars_cc = [u_0, u_1, lmbda, V_cell]
        soln_vars_cv = [u_0, u_1]

        if cycler.current_mode_type in (gitt, galvanostatic):
            soln_vars = soln_vars_cc
        elif cycler.current_mode_type in (potentiostatic, cyclic_voltammetry):
            soln_vars = soln_vars_cv
        else:
            raise ValueError("Unknown cycling mode")

        problem_t0 = solvers.NonlinearPDE_SNESProblem(F2D, J2D, soln_vars, bcs, P=J2D)
        snes.setFunction(problem_t0.F_block, Fvec2d)
        snes.setJacobian(problem_t0.J_block, J=Jmat2d, P=Jmat2d)
        x2d = fem.petsc.create_vector(F2D, kind="mpi")
        x2d.set(0.0)
        t0 = time.time()
        snes.solve(None, x2d)
        t1 = time.time()
        snes.destroy()
        Jmat2d.destroy()
        Fvec2d.destroy()
        x2d.destroy()
        petsc_options.clear()
        # if cycler.current_mode_type in (gitt, galvanostatic):
        #     PETSc.Sys.Print("V_cell (initial guess) [V]:", f"{V_cell.x.array[0] * ref["phi"]:.3f}")
        # else:
        #     PETSc.Sys.Print("V_cell (prescribed) [V]:", f"{cycler.cv_voltage_function(0):.3f}")

        u_avg_left_tilde = comm.allreduce(fem.assemble_scalar(fem.form(u_0 * ds(markers.left),
                                                                        entity_maps=entity_maps)), op=MPI.SUM) / A_left_tilde
        u_avg_left = u_avg_left_tilde * phi_ref
        u_stdev_left_tilde = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(
                                            (u_0 - u_avg_left_tilde) ** 2 * ds(markers.left),
                                            entity_maps=entity_maps)), op=MPI.SUM) / A_left_tilde)
        u_stdev_left = u_stdev_left_tilde  * phi_ref

        u_avg_right_tilde = comm.allreduce(fem.assemble_scalar(fem.form(u_1 * ds(markers.right),
                                                                        entity_maps=entity_maps)), op=MPI.SUM) / A_right_tilde
        u_avg_right = u_avg_right_tilde * phi_ref
        u_stdev_right_tilde = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(
                                            (u_1 - u_avg_right_tilde) ** 2 * ds(markers.right),
                                            entity_maps=entity_maps)), op=MPI.SUM) / A_right_tilde)
        u_stdev_right = u_stdev_right_tilde  * phi_ref
        I_right = comm.allreduce(fem.assemble_scalar(fem.form(
                                inner(kappa_pos_am * phi_ref * L_ref ** (k) * grad(u_1), n) * ds(markers.right),
                                entity_maps=entity_maps)), op=MPI.SUM)
        eta_avg = phi_ref / A_se_am_tilde * comm.allreduce(fem.assemble_scalar(fem.form((u_r - u_l - U_ocp(c_r)) * dInterface,
                                                                        entity_maps=entity_maps)), op=MPI.SUM)
        V_ocp_avg = phi_ref / A_se_am_tilde * comm.allreduce(fem.assemble_scalar(fem.form(U_ocp(c_r) * dInterface,
                                                                        entity_maps=entity_maps)), op=MPI.SUM)
        if comm_rank == 0:
            stats_writer.writerow(
                         {
                        "t [s]": 0,
                        "I left [A]": np.nan,
                        "I interface [A]": np.nan,
                        "I interface (potential left) [A]": np.nan,
                        "I interface (potential right) [A]": np.nan,
                        "I (butler-volmer) [A]": np.nan,
                        "I right [A]": I_right,
                        "I (target) right [A]": np.nan,
                        "u (avg) left [V]": u_avg_left,
                        "u (stdev) left [v]": u_stdev_left,
                        "u (avg) right [V]": u_avg_right,
                        "u (stdev) right [v]": u_stdev_right,
                        "surface overpotential (avg) [V]": eta_avg,
                        "V (ocp) (avg) [V]": V_ocp_avg,
                        "i (avg) left [A/m2]": np.nan,
                        "i (stdev) left [A/m2]": np.nan,
                        "i (avg) se/am [A/m2]": np.nan,
                        "i (stdev) se/am [A/m2]": np.nan,
                        "i (avg) right [A/m2]": np.nan,
                        "i (stdev) right [A/m2]": np.nan,
                        "c surf (avg) (normalized)": np.nan,
                        "c surf (stdev) (normalized)": np.nan,
                        "c (avg) (normalized)": c_init,
                        "I_interface error norm (normalized)": np.nan,
                        "I_interface error norm c (normalized)": np.nan,
                        "Diffusivity [m2/s]": args.D,
                        "Positive Wa": args.Wa_p,
                        "Kr": args.kr,
                })
            fp.flush()

        PETSc.Sys.Print(f"Finished computation of initial (t = 0) potential distribution!\nn_dofs: {n_dofs_t0:,}\nsolve time: {t1 - t0:.3f}s")
        PETSc.Sys.Print("************Solve for Improved Guess for Concentration Distribution*******************")
        petsc_options.clear()
        n_dofs_c = VC_map.size_global*VC.dofmap.index_map_bs
        u_int.interpolate(u_1)
        F_c = (c - c0)/dt * q * dx_c + inner(ufl.grad(c), ufl.grad(q)) * dx_c
        F_2 += -inner(kappa_pos_am * phi_ref/(D * faraday_const * c_ref) * grad(u_int), n_1) * q * ds_c(markers.electrolyte_v_positive_am)
        # F_c += -inner(grad(u_int), n_1) * q * ds_c(markers.electrolyte_v_positive_am)
        problem_c = fem.petsc.NonlinearProblem(F_c, c, bcs=[])
        solver = petsc_nls.NewtonSolver(comm, problem_c)
        solver.convergence_criterion = "residual"
        solver.maximum_iterations = 100
        solver.rtol = 1e-8

        ksp = solver.krylov_solver
        option_prefix = ksp.getOptionsPrefix()
        petsc_options[f"{option_prefix}ksp_type"] = "cg"
        petsc_options[f"{option_prefix}pc_type"] = args.amg_type
        for optk, optv in solver_params.AMG_TYPES[args.amg_type].items():
                petsc_options[f"{option_prefix}{optk}"] = optv
        # petsc_options[f"{option_prefix}pc_factor_levels"] = 0
        # petsc_options[f"{option_prefix}pc_factor_fill"] = 2.0
        ksp.setFromOptions()
        t0 = time.time()
        n_iters, converged = solver.solve(c)
        t1 = time.time()
        PETSc.Sys.Print(f"Finished computation of improved guess of concentration distribution!\nn_dofs: {n_dofs_c:,}\nsolve time: {t1 - t0:.3f}s")
        PETSc.Sys.Print(utils.starpad("*"))
    ########################################################################################################################################

    # interpolate
    V = fem.functionspace(domain, ("DG", 1))
    u = fem.Function(V)

    # current density distribution setup
    W = fem.functionspace(submesh_positive_am, ("CG", 1, (tdim,)))
    current_expr = fem.Expression(-kappa_pos_am * phi_ref/L_ref * ufl.grad(u_1), W.element.interpolation_points)
    current_h = fem.Function(W, name='current_density')
    tol_fun = fem.Function(V1)
    tol_fun_left = fem.Function(V1)
    tol_fun_right = fem.Function(V1)

    idx = 0
    stop = False

    cvtx = io.VTXWriter(comm, concentration_file, [c], engine="BP5")
    u_vtx = io.VTXWriter(comm, output_potential_file, [u], engine="BP5")
    u_vtx.write(0)

    # cycler.next()
    dt.value = cycler.dt
    while not cycler.stop:
        if cycler.current_mode_type == cyclic_voltammetry:
            u_right.x.array[:] = cycler.cv_voltage_function(0)/phi_ref
            bc_right = fem.dirichletbc(
                                       u_right, fem.locate_dofs_topological(u_1.function_space, fdim, ft_positive_am.find(markers.right)))
            bcs = [bc_left, bc_right]
            soln_vars = [u_0, u_1, c]
        if cycler.current_mode_type == gitt:
            I_tot.value = utils.get_c_rate_current(c_max, cycler.gitt_current_function(0), vol_pos_am)
            I_tot_tilde.value = I_tot.value /(L_ref ** (k) * kappa_total * phi_ref)
            soln_vars = [u_0, u_1, lmbda, V_cell, c]
            bcs = [bc_left]
        if cycler.current_mode_type == galvanostatic:
            I_tot.value = cycler.current_mode["direction"] * utils.get_c_rate_current(c_max, cycler.current_mode["c-rate"], vol_pos_am)
            I_tot_tilde.value = I_tot.value /(L_ref ** (k) * kappa_total * phi_ref)
            soln_vars = [u_0, u_1, lmbda, V_cell, c]
            bcs = [bc_left]
        elif cycler.current_mode_type == potentiostatic:
            u_right.x.array[:] = cycler.current_mode["voltage"]/phi_ref
            bc_right = fem.dirichletbc(
                                       u_right, fem.locate_dofs_topological(u_1.function_space, fdim, ft_positive_am.find(markers.right)))
            bcs = [bc_left, bc_right]
            soln_vars = [u_0, u_1, c]

        PETSc.Sys.Print(f"Time: {cycler.time*t_ref:,.1f}s\n")
        petsc_options.clear()
        if cycler.current_mode_type in (galvanostatic, gitt):
            J = J_cc
            F = F_cc
            P = J
            Jmat = fem.petsc.create_matrix(J, kind="nest")
            nested_IS = Jmat.getNestISs()
            IS_u0 = nested_IS[0][0]
            IS_u1 = nested_IS[0][1]
            IS_l = nested_IS[0][2]
            IS_v = nested_IS[0][3]
            IS_c = nested_IS[0][4]
            IS_u = IS_u0.sum(IS_u1)
            IS_ulg = IS_u.sum(IS_l).sum(IS_v)
        elif cycler.current_mode_type in (potentiostatic, cyclic_voltammetry):
            J = J_cv
            F = F_cv
            P = J
            Jmat = fem.petsc.create_matrix(J, kind="nest")
            nested_IS = Jmat.getNestISs()
            IS_u0 = nested_IS[0][0]
            IS_u1 = nested_IS[0][1]
            IS_c = nested_IS[0][2]
            IS_ulg = IS_u0.sum(IS_u1)

        Jmat = fem.petsc.create_matrix(J, kind="mpi")
        Pmat = fem.petsc.create_matrix(P, kind="mpi")
        Fvec = fem.petsc.create_vector(F, kind="mpi")
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
        snes.getKSP().getPC().setFieldSplitIS(("un", IS_ulg), ("c", IS_c))
        petsc_options = PETSc.Options()
        petsc_options[f'{snes.getKSP().getOptionsPrefix()}ksp_gmres_restart'] = 100
        for kopt, vopt in solver_params.LINESEARCH.items():
            petsc_options[kopt] = vopt

        petsc_options['log_view'] = None

        petsc_options[f"{snes.getKSP().getOptionsPrefix()}pc_fieldsplit_off_diag_use_amat"] = True
        petsc_options[f"{snes.getKSP().getOptionsPrefix()}pc_fieldsplit_detect_saddle_point"] = True

        ksp_u, ksp_c = snes.getKSP().getPC().getFieldSplitSubKSP()

        snes.getKSP().getPC().setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        snes.getKSP().getPC().setFieldSplitSchurPreType(PETSc.PC.SchurPreType.SELFP)
        snes.getKSP().getPC().setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)
        ksp_u.setType(PETSc.KSP.Type.FGMRES)
        ksp_u.getPC().setType(PETSc.PC.Type.ILU)
        ksp_u.setTolerances(rtol=1e-7, max_it=1000)
        petsc_options[f"{ksp_u.getOptionsPrefix()}pc_factor_levels"] = 0
        petsc_options[f"{ksp_u.getOptionsPrefix()}pc_factor_fill"] = 2.0

        ksp_c.setType(PETSc.KSP.Type.CG)
        ksp_c.getPC().setType(args.amg_type)
        ksp_c.setTolerances(rtol=1e-7, max_it=1000)

        petsc_options[f"{ksp_c.getOptionsPrefix()}mat_schur_complement_ainv_type"] = "lump"
        petsc_options[f"{ksp_c.getOptionsPrefix()}inner_ksp_type"] = "preonly"
        petsc_options[f"{ksp_c.getOptionsPrefix()}inner_pc_type"] = "ilu"
        petsc_options[f"{ksp_c.getOptionsPrefix()}inner_pc_factor_levels"] = 0
        petsc_options[f"{ksp_c.getOptionsPrefix()}inner_pc_factor_fill"] = 2.0
        petsc_options[f"{ksp_c.getOptionsPrefix()}upper_ksp_type"] = "preonly"
        petsc_options[f"{ksp_c.getOptionsPrefix()}upper_pc_type"] = "ilu"
        petsc_options[f"{ksp_c.getOptionsPrefix()}upper_pc_factor_levels"] = 0
        petsc_options[f"{ksp_c.getOptionsPrefix()}upper_pc_factor_fill"] = 2.0

        for optk, optv in solver_params.AMG_TYPES[args.amg_type].items():
            petsc_options[f"{ksp_c.getOptionsPrefix()}{optk}"] = optv

        ksp_u.setFromOptions()
        ksp_c.setFromOptions()
        snes.getKSP().setFromOptions()

        problem = solvers.NonlinearPDE_SNESProblem(F, J, soln_vars, bcs, P=P)
        snes.setFunction(problem.F_block, Fvec)
        snes.setJacobian(problem.J_block, J=Jmat, P=Pmat)
        snes.setFromOptions()
        snes.view()

        x = fem.petsc.create_vector(F, kind="mpi")
        x.set(0.0)
        PETSc.Log().begin()
        t0 = time.time()
        snes.solve(None, x)
        t1 = time.time()
        PETSc.Sys.Print(f"SNES converged reason: {snes.getConvergedReason()}, solve time: {t1-t0:.3f}s")
        PETSc.Log().view(log_viewer)
        # increase step time after 5 seconds
        if t_ref * cycler.time >= 5:
            cycler._dt = 5 * args.dt
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
        c0.x.array[:] = c.x.array
        current_h.interpolate(current_expr)
        I_left = comm.allreduce(fem.assemble_scalar(fem.form(
                                inner(kappa_elec * phi_ref * L_ref ** (k) * grad(u_0), n) * ds(markers.left),
                                entity_maps=entity_maps)), op=MPI.SUM)
        I_right = comm.allreduce(fem.assemble_scalar(fem.form(
                                inner(kappa_pos_am * phi_ref * L_ref ** (k) * grad(u_1), n) * ds(markers.right),
                                entity_maps=entity_maps)), op=MPI.SUM)
        I_interface = comm.allreduce(fem.assemble_scalar(fem.form(
                                inner(faraday_const * D * c_ref * L_ref ** (k) * grad(c(r_res)), n_r) * dInterface,
                                entity_maps=entity_maps)), op=MPI.SUM)

        I_interface_l = comm.allreduce(fem.assemble_scalar(fem.form(
                        inner(kappa_elec * phi_ref * L_ref ** (k) * grad(u_l), n_l) * dInterface,
                        entity_maps=entity_maps)), op=MPI.SUM)

        I_interface_r = comm.allreduce(fem.assemble_scalar(fem.form(
                inner(kappa_pos_am * phi_ref * L_ref ** (k) * grad(u_r), n_r) * dInterface,
                entity_maps=entity_maps)), op=MPI.SUM)

        I_interface_error = comm.allreduce(fem.assemble_scalar(fem.form(
                        phi_ref * L_ref ** (k) * np.abs(inner(kappa_elec * grad(u_l), n_l) + inner(kappa_pos_am * grad(u_r), n_r)) * dInterface,
                        entity_maps=entity_maps)), op=MPI.SUM)

        I_interface_error_sq = comm.allreduce(fem.assemble_scalar(fem.form(
                        (phi_ref * L_ref ** (-k) * (inner(kappa_elec * grad(u_l), n_l) + inner(kappa_pos_am * grad(u_r), n_r))) ** 2 * L_ref ** 2 * dInterface,
                        entity_maps=entity_maps)), op=MPI.SUM)

        error = phi_ref * L_ref ** (-k) * (inner(kappa_elec * grad(u_l), n_l) + inner(kappa_pos_am * grad(u_r), n_r))
        error_c = L_ref ** (-k) * (phi_ref/2 * inner(kappa_elec * grad(u_l) + kappa_pos_am * grad(u_r), n_r) - inner(c_ref * faraday_const * D * grad(c_r), n_r))
        i_x_l = inner(kappa_elec * grad(u_l), n_l) * phi_ref * L_ref ** (-k)
        i_x_r = inner(kappa_pos_am * grad(u_r), n_r) * phi_ref * L_ref ** (-k)
        I_x_norm_l = np.sqrt(comm.allreduce(fem.assemble_scalar(
                                    fem.form(inner(i_x_l, i_x_l) * L_ref ** 2 * dInterface, entity_maps=entity_maps)), op=MPI.SUM))
        I_x_norm_r = np.sqrt(comm.allreduce(fem.assemble_scalar(
                                    fem.form(inner(i_x_r, i_x_r) * L_ref ** 2 * dInterface, entity_maps=entity_maps)), op=MPI.SUM))
        I_x_norm = 0.5 * (I_x_norm_l + I_x_norm_r)

        I_interface_error_norm = np.sqrt(comm.allreduce(fem.assemble_scalar(
                                    fem.form(inner(error, error) * L_ref ** (k+1) * dInterface, entity_maps=entity_maps)), op=MPI.SUM))
        I_interface_error_norm_c = np.sqrt(comm.allreduce(fem.assemble_scalar(
                                    fem.form(inner(error_c, error_c) * L_ref ** (k+1) * dInterface, entity_maps=entity_maps)), op=MPI.SUM))
        u_avg_left_tilde = comm.allreduce(fem.assemble_scalar(fem.form(u_0 * ds(markers.left),
                                                                        entity_maps=entity_maps)), op=MPI.SUM) / A_left_tilde
        u_avg_left = u_avg_left_tilde * phi_ref
        u_stdev_left_tilde = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(
                                            (u_0 - u_avg_left_tilde) ** 2 * ds(markers.left),
                                            entity_maps=entity_maps)), op=MPI.SUM) / A_left_tilde)
        u_stdev_left = u_stdev_left_tilde  * phi_ref

        u_avg_right_tilde = comm.allreduce(fem.assemble_scalar(fem.form(u_1 * ds(markers.right),
                                                                        entity_maps=entity_maps)), op=MPI.SUM) / A_right_tilde
        c_surf_avg_tilde = comm.allreduce(fem.assemble_scalar(fem.form(c_r * dInterface,
                                                                        entity_maps=entity_maps)), op=MPI.SUM) / A_se_am_tilde
        c_surf_avg = c_surf_avg_tilde
        c_surf_stdev = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(
                                            (c_r - c_surf_avg_tilde) ** 2 * dInterface,
                                            entity_maps=entity_maps)), op=MPI.SUM) / A_se_am_tilde)
        c_avg_tilde = 1/vol_pos_am_tilde * comm.allreduce(fem.assemble_scalar(fem.form(c * dx(markers.positive_am),
                                                                        entity_maps=entity_maps)), op=MPI.SUM)

        u_avg_right = u_avg_right_tilde * phi_ref
        u_stdev_right_tilde = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(
                                            (u_1 - u_avg_right_tilde) ** 2 * ds(markers.right),
                                            entity_maps=entity_maps)), op=MPI.SUM) / A_right_tilde)
        u_stdev_right = u_stdev_right_tilde  * phi_ref
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
        eta_avg = phi_ref / A_se_am_tilde * comm.allreduce(fem.assemble_scalar(fem.form((u_r - u_l - U_ocp(c_r)) * dInterface,
                                                                        entity_maps=entity_maps)), op=MPI.SUM)
        I_bv = L_ref ** (k+1) * comm.allreduce(fem.assemble_scalar(fem.form(2*i0_p * (ufl.sinh(0.5 * phi_ref * (u_r - u_l - U_ocp(c_r)) * faraday_const / (R * T))) * dInterface, entity_maps=entity_maps)), op=MPI.SUM)
        V_ocp_avg = phi_ref / A_se_am_tilde * comm.allreduce(fem.assemble_scalar(fem.form(U_ocp(c_r) * dInterface,
                                                                        entity_maps=entity_maps)), op=MPI.SUM)
        dt.value = cycler.dt
        cycler.check_stop_criteria(I_cell=np.abs(I_right), V_cell=u_avg_right)
        cvtx.write(cycler.time)

        u.interpolate(u_0, cells1=submesh_electrolyte_to_mesh, cells0=np.arange(len(submesh_electrolyte_to_mesh)))
        u.interpolate(u_1, cells1=submesh_positive_am_to_mesh, cells0=np.arange(len(submesh_positive_am_to_mesh)))
        u.x.scatter_forward()
        u_vtx.write(cycler.time)

        # current density distribution
        i_intervals = np.linspace(0, 1.05 * np.max([np.abs(i_avg_left), np.abs(i_avg_right)]), 101)
        densities = current_density_distribution(comm, current_h(r_res), n_r, tol_fun_left, tol_fun_right, dInterface, entity_maps, i_intervals)
        densities[:, 2] /= A_se_am_tilde
        if comm_rank == 0 and args.plot:
            fig, ax = plt.subplots()
            ax.bar(0.5*(densities[:, 0] + densities[:, 1]), densities[:, 2], width=i_intervals[1], align="center")
            # ax.plot(densities[:, 1], densities[:, 2])
            ax.set_box_aspect(1)
            ax.set_xlabel(r"i [A/m$^2$]")
            ax.set_ylabel("relative areal density")
            ax.set_ylim([0, 1.01 * np.max(densities[:, 2])])
            ax.set_xlim([0, np.max(i_intervals)])
            plt.tight_layout()
            plt.savefig(i_interface_density_plot, )
            # plt.show()

        if comm_rank == 0:
            stats_writer.writerow(
                         {
                         "t [s]": cycler.time * t_ref,
                        "I left [A]": I_left,
                        "I interface [A]": I_interface,
                        "I interface (potential left) [A]": I_interface_l,
                        "I interface (potential right) [A]": I_interface_r,
                        "I (butler-volmer) [A]": I_bv,
                        "I right [A]": I_right,
                        "I (target) right [A]": I_tot.value,
                        "u (avg) left [V]": u_avg_left,
                        "u (stdev) left [v]": u_stdev_left,
                        "u (avg) right [V]": u_avg_right,
                        "u (stdev) right [v]": u_stdev_right,
                        "surface overpotential (avg) [V]": eta_avg,
                        "V (ocp) (avg) [V]": V_ocp_avg,
                        "i (avg) left [A/m2]": i_avg_left,
                        "i (stdev) left [A/m2]": i_stdev_left,
                        "i (avg) se/am [A/m2]": i_avg_se_am,
                        "i (stdev) se/am [A/m2]": i_stdev_se_am,
                        "i (avg) right [A/m2]": i_avg_right,
                        "i (stdev) right [A/m2]": i_stdev_right,
                        "c surf (avg) (normalized)": c_surf_avg,
                        "c surf (stdev) (normalized)": c_surf_stdev,
                        "c (avg) (normalized)": c_avg_tilde,
                        "I_interface error norm (normalized)": I_interface_error_norm / I_x_norm,
                        "I_interface error norm c (normalized)": I_interface_error_norm_c / I_x_norm,
                        "Diffusivity [m2/s]": args.D,
                        "Positive Wa": args.Wa_p,
                        "Kr": args.kr,
                })
            fp.flush()
        cycler.next()
    cvtx.close()
    fp.close()

    time_elapsed = timeit.default_timer() - start_time

    metadata = {
        "I left [A]": I_left,
        "I interface [A]": I_interface,
        "I right [A]": I_right,
        "I (target) right [A]": I_tot_,
        "I interface (electrolyte potential) [A]": I_interface_l,
        "I interface (active material potential) [A]": I_interface_r,
        "I_interface error (potential) [A]": I_interface_error,
        "I_interface error norm (normalized)": I_interface_error_norm / I_x_norm,
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
        "concentration field polynomial degree (p)": args.p_concentration,
        "SE potential field polynomial degree (p)": args.p_u0,
        "AM potential field polynomial degree (p)": args.p_u1,
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
        axis = directions[args.transport_direction.lower()]
        points[axis:, ] = z

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

            c_vals = all_c_vals[all_c_vals[:, axis].argsort()]
            u_vals = all_u_vals[all_u_vals[:, axis].argsort()]
            t_str = f"{cycler.time*t_ref:.3f}"

            c_json = {
                "t": cycler.time * t_ref,
                "Wa": args.Wa_p,
                "D": args.D,
                "kr": args.kr,
                "x": c_vals[:, axis].tolist(),
                "y": (c_vals[:, 3]*c_ref).tolist()
            }

            p_json = {
                "t": cycler.time * t_ref,
                "Wa": args.Wa_p,
                "D": args.D,
                "kr": args.kr,
                "x": u_vals[:, axis].tolist(),
                "y": u_vals[:, 3].tolist()
            }

            potential_json_path = os.path.join(results_dir, f"potential-{t_str}.json")
            concentration_json_path = os.path.join(results_dir, f"concentration-{t_str}.json")

            with open(potential_json_path, "w", encoding='utf-8') as f:
                json.dump(p_json, f, ensure_ascii=False, indent=4)

            with open(concentration_json_path, "w", encoding='utf-8') as f:
                json.dump(c_json, f, ensure_ascii=False, indent=4)

            fig, ax = plt.subplots()
            ax.plot(c_vals[:, axis], c_vals[:, 3]*c_ref, 'k', label=r'0.5$L_x$,0.5$L_y$', linewidth=1)
            ax.grid(True)
            ax.legend()
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_box_aspect(1)
            ax.set_ylabel(r'$\hat{c}$', rotation=90, labelpad=0, fontsize='xx-large')
            ax.set_xlabel(r'$\hat{x}$')
            ax.set_title(r'$\mathrm{Wa}$ = ' + f'{args.Wa_p}' + ',' + r'$\frac{\kappa}{\sigma}$ = ' + f'{args.kr}' + f' t = {cycler.time*t_ref:.3f}s')
            plt.tight_layout()
            plt.savefig(concentration_plot_file.replace(".eps", f"{t_str}.eps"))

            fig, ax = plt.subplots()
            ax.plot(u_vals[:, axis], u_vals[:, 3], 'k', label=r'0.5$L_x$,0.5$L_y$', linewidth=1)
            ax.grid(True)
            ax.legend()
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_box_aspect(1)
            ax.set_ylabel(r'$\hat{\phi}$', rotation=90, labelpad=0, fontsize='xx-large')
            ax.set_xlabel(r'$\hat{x}$')
            ax.set_title(r'$\mathrm{Wa}$ = ' + f'{args.Wa_p}' + ',' + r'$\frac{\kappa}{\sigma}$ = ' + f'{args.kr}' + f' t = {cycler.time*t_ref:.3f}s')
            plt.tight_layout()
            plt.savefig(potential_plot_file.replace(".eps", f"{t_str}.eps"))
