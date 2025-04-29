#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import matspy

import plot_opts, solvers, utils
plt.rcParams.update(plot_opts.params)

class ShuntCurrentsParameters:
    def __init__(self, N_s=100, d_p=0.01, V_cell=1.0, kappa=4.0, H_p=0.03,
                 A_m=0.006, L_p=0.02, a=100, i0=10, a_a=0.5, a_c=0.5):
        self._N_s = N_s
        self._d_p = d_p
        self._V_cell = V_cell
        self._kappa = kappa
        self._H_p = H_p
        self._A_m = A_m
        self._L_p = L_p
        self._a = a
        self._i0 = i0
        self._a_a = a_a
        self._a_c = a_c
        self._eta_s_0 = None
        self._faraday_constant = 96485
        self._R = 8.314
        self._T = 298

    @property
    def N_s(self):
        return self._N_s

    @property
    def d_p(self):
        return self._d_p

    @property
    def V_cell(self):
        return self._V_cell

    @property
    def kappa(self):
        return self._kappa

    @property
    def H_p(self):
        return self._H_p

    @property
    def A_m(self):
        return self._A_m

    @property
    def L_p(self):
        return self._L_p

    @property
    def a(self):
        return self._a

    @property
    def i0(self):
        return self._i0

    @property
    def a_a(self):
        return self._a_a

    @property
    def a_c(self):
        return self._a_c

    @property
    def L(self):
        return 0.5 * self.N_s * self.d_p

    @property
    def R_p(self):
        return self.L_p / self.kappa

    @property
    def faraday_constant(self):
        return self._faraday_constant

    @property
    def R(self):
        return self._R

    @property
    def T(self):
        return self._T

    @property
    def eta_s_0(self):
        return self._eta_s_0

    @property
    def omega(self):
        return np.sqrt(self.a * self.i0 * (self.a_a + self.a_c) * self.faraday_constant / self.kappa / self.R / self.T)


def i_p(eta, p):
    return np.sqrt(2 * p.a * p.i0 * p.kappa * R * T/(p.a_a * p.a_c * F)) *\
        np.sqrt(p.a_c * np.exp(p.a_a * F * eta/(R * T)) +\
                p.a_a * np.exp(-p.a_c * F * eta/(R * T)) - p.a_a - p.a_c)


def i_p_prime(eta, p):
    return np.sqrt(2 * p.a * p.i0 * p.kappa * R * T/(p.a_a * p.a_c * F)) * (p.a_a * p.a_c * F / (R * T) ) * (np.exp(p.a_a * F * eta/(R * T)) -\
                np.exp(-p.a_c * F * eta/(R * T)))/\
            np.sqrt(p.a_c * np.exp(p.a_a * F * eta/(R * T)) + p.a_a * np.exp(-p.a_c * F * eta/(R * T)) - p.a_a - p.a_c)


def source(y, eta_s0, p, kinetics_type="linear"):
    if kinetics_type == "linear":
        return (p.V_cell / p.d_p * y)
    elif kinetics_type == "butler_volmer":
        return (p.V_cell / p.d_p * y - eta_s0 + i_p(eta_s0, p)/i_p_prime(eta_s0, p))
    raise ValueError("Unknown kinetics type")


def lambda_squared(eta_s0, p, kinetics_type="linear"):
    if kinetics_type == "linear":
        return p.H_p / (p.A_m * (p.L_p + 1/p.omega))
    elif kinetics_type == "butler_volmer":
        return p.H_p / (p.kappa * p.A_m) * i_p_prime(eta_s0, p) / (i_p_prime(eta_s0, p) * p.R_p + 1)
    raise ValueError("Unknown kinetics type")


def solve_for_manifold_potential(eta_s0, p, N, h, kinetics_type="linear"):
    """
    Center finite difference discretization of of u`` + l^2 * u = l^2 * f(y)
    with bc:
        u(y=0) = 0
        k * du/dy(y=L) = 0
    """
    A = np.zeros((N, N))
    b = np.zeros((N, 1))
    for idx in range(N):
        l2 = 0
        if idx < N:
            l2 = lambda_squared(eta_s0[idx], p, kinetics_type=kinetics_type)
        if idx < N - 1:
            b[idx] = -l2 * source((idx+1) * h, eta_s0[idx], p, kinetics_type=kinetics_type)
        if idx == 0:  # y = h
            A[idx, idx+1] = 1/h**2
            A[idx, idx] = -2/h**2 - l2
        elif idx == N-1:  # y = L + h
            A[idx, idx] = 1
            A[idx, idx-1] = -1
        else:
            A[idx, idx+1] = 1/h**2
            A[idx, idx] = -2/h**2 - l2
            A[idx, idx-1] = 1/h**2
    u = np.linalg.solve(A, b)

    return A, b, u


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='secondary current distribution')
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    # parser.add_argument('--study_name', help='identify for groups of varied parameters', required=True)
    parser.add_argument("-N_s", "--N_s", help="number of cells in stack", nargs='?', const=1, default=100, type=int)
    parser.add_argument("-L_p", "--L_p", help="Length [m]", nargs='?', const=1, default=0.02, type=float)
    parser.add_argument("-kappa", "--kappa", help="Conductivity [S/m]", nargs='?', const=1, default=4, type=float)
    parser.add_argument("-w", "--w", help="reaction penetration", nargs='?', const=1, default=100.0, type=float)
    parser.add_argument("-a", "--a", help="specific area [1/m]", nargs='?', const=1, default=100.0, type=float)
    parser.add_argument("-vary", "--vary", help="variable under study", nargs='?', const=1, default="L_p", type=str)
    parser.add_argument("--show_plot", help="whether to display plot", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    workdir = args.mesh_folder
    results_dir = os.path.join(workdir, args.vary)
    utils.make_dir_if_missing(results_dir)
    utils.make_dir_if_missing(os.path.join(results_dir, "potential"))
    R = 8.314
    T = 298
    F = 96485
    omega = args.w
    a = args.a
    kappa = args.kappa
    a_a = 0.5 #args.a_a
    a_c = 0.5 #args.a_c
    i0 = args.kappa * R * T * omega **2 / (F * a * (a_a + a_c))
    p = ShuntCurrentsParameters(a=a, kappa=kappa, a_a=a_a, a_c=a_c, i0=i0)
    N = 500
    h = p.N_s * p.d_p / 2 / N
    y = np.zeros((N+1, 1))
    for idx in range(N):
        y[idx] = idx * h
    eta_s0 = 1e-8 * np.ones((N+1, 1))
    _, _, u_lin = solve_for_manifold_potential(eta_s0, p, N+1, h, kinetics_type="linear")
    _, _, u_bv = solve_for_manifold_potential(eta_s0, p, N+1, h, kinetics_type="butler_volmer")
    var_value = 0
    if args.vary == "L_p":
        var_value = p.L_p
    elif args.vary == "N_s":
        var_value = p.N_s
    elif args.vary == "w":
        var_value = p.omega
    else:
        raise ValueError("Unknown study type")
    fig, ax = plt.subplots()
    ax.plot(y[:-1], u_lin[:-1], label="Linear")
    ax.plot(y[:-1], u_bv[:-1], label="Butler-Volmer")
    ax.plot([0, p.L], [0, 50], linestyle='--', color='cyan', label="Electrode potential")
    ax.set_xlim([0, p.L])
    ax.set_ylim([0, 50])
    ax.grid()
    ax.set_box_aspect(1)
    ax.legend()
    plt.tight_layout()
    if args.show_plot:
        plt.show()
    else:
        plt.savefig(os.path.join(results_dir, "potential", f"{var_value}.eps"))
