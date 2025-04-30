#!/usr/bin/env python3
import argparse
import logging
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from mpi4py import MPI
import matspy
import warnings

warnings.simplefilter("ignore")

import plot_opts, solvers, utils
plt.rcParams.update(plot_opts.params)
logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)
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


def vary_L_p():
    return np.linspace(0.005, 0.05, 10)


def vary_N_s():
    return np.linspace(20, 100, 9)

def vary_omega():
    return [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

def I_manifold(u_bv, p):
    N = u_bv.shape[0]
    grad_u = np.zeros((N - 1))
    for idx in range(1, N-1):
        grad_u[idx] = (u_bv[idx+1] - u_bv[idx - 1]) / (2 * h)
    I_m = -p.kappa * p.A_m * grad_u
    return I_m


def i_port_approx(u_bv, p):
    N = u_bv.shape[0]
    laplacian_u = np.zeros((N - 1, 1))
    for idx in range(1, N-1):
        laplacian_u[idx] = (u_bv[idx+1] -2 * u_bv[idx] + u_bv[idx - 1]) / (h ** 2)
    i_p_approx = -p.kappa * p.A_m / p.H_p * laplacian_u
    return i_p_approx


def solve_loop(N, h, p, eta_s0, tol, max_its):
    error = tol + 1
    its = 0
    _, _, u_lin = solve_for_manifold_potential(eta_s0, p, N+1, h, kinetics_type="linear")
    eta_s1 = np.zeros(eta_s0.shape)
    y = np.zeros((N+1, 1))
    while error > tol and its < max_its:
        its += 1
        _, _, u_bv = solve_for_manifold_potential(eta_s0, p, N+1, h, kinetics_type="butler_volmer")
        ip_approx = i_port_approx(u_bv, p)
        eta_s1[:-1] = (p.V_cell/p.d_p * y[:-1] - ip_approx * p.R_p - u_bv[1:])/p.N_s
        error = np.linalg.norm(eta_s1 - eta_s0)
        eta_s0 = eta_s1
    print(f"Error: {error:.0e}, Tolerance: {tol:.0e}, Iterations: {its}")
    return u_lin, u_bv


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
    utils.make_dir_if_missing(os.path.join(results_dir, "i_port"))
    utils.make_dir_if_missing(os.path.join(results_dir, "manifold"))
    max_its = 10
    tol = 1e-8
    R = 8.314
    T = 298
    F = 96485
    a_a = 0.5
    a_c = 0.5
    a = args.a
    kappa = args.kappa
    omega = 100
    var_value = 0
    N = 50
    variables = vary_omega()
    I_m_max_vals_bv = []
    I_m_max_vals_lin = []
    I_ds_vals_bv = []
    I_ds_vals_lin = []
    i_p_max_vals_bv = []
    i_p_max_vals_lin = []
    if args.vary == 'w':
        for omega in variables:
            i0 = args.kappa * R * T * omega **2 / (F * a * (a_a + a_c))
            p = ShuntCurrentsParameters(a=a, kappa=kappa, a_a=a_a, a_c=a_c, i0=i0)
            h = p.N_s * p.d_p / 2 / N
            y = np.zeros((N+1, 1))
            for idx in range(N):
                y[idx] = idx * h
            eta_s0 = 1e-8 * np.ones((N+1, 1))
            u_lin, u_bv = solve_loop(N, h, p, eta_s0, tol=tol, max_its=max_its)

            if args.vary == "L_p":
                var_value = p.L_p
            elif args.vary == "N_s":
                var_value = p.N_s
            elif args.vary == "w":
                var_value = p.omega
            else:
                raise ValueError("Unknown study type")
            
            fig, ax = plt.subplots()
            ax.plot(0.5*p.N_s * y[:-1]/p.L, u_bv[:-1], 'r-.', label="Butler-Volmer")
            ax.plot(0.5*p.N_s * y[:-1]/p.L, u_lin[:-1], 'b-', label="Linear")
            ax.plot([0, 0.5 * p.N_s], [0, 50], linestyle='--', color='black', label="Electrode potential")
            ax.set_xlim([0, 0.5 * p.N_s])
            ax.set_ylim([0, 50])
            ax.set_xlabel("Cell number")
            ax.set_ylabel("Potential [V]")
            ax.grid(color="cyan", linewidth=0.1)
            ax.set_box_aspect(1)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "potential", f"{var_value}.eps"))
            plt.close()

            fig, ax = plt.subplots()
            port_current_density_bv = i_port_approx(u_bv, p)
            port_current_density_lin = i_port_approx(u_lin, p)
            ax.plot(0.5*p.N_s * y[:-1]/p.L, port_current_density_bv, 'r-.', linewidth=0.5, label="Butler-Volmer")
            ax.plot(0.5*p.N_s * y[:-1]/p.L, port_current_density_lin, 'b', linewidth=0.25, label="Linear")
            ax.grid(color='cyan', linewidth=0.1)
            ax.set_xlim([0, 0.5*p.N_s])
            ax.set_ylim([0, 1.01 * np.max(port_current_density_bv)])
            ax.set_box_aspect(1)
            ax.set_ylabel(r"$i_p$ [A/m$^2$]")
            ax.set_xlabel("Cell number")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "i_port", f"{var_value}.eps"))
            plt.close()
            I_m_bv = I_manifold(u_bv, p)
            I_m_lin = I_manifold(u_lin, p)
            I_m_max_bv = np.max(np.abs(I_m_bv))
            I_m_max_lin = np.max(np.abs(I_m_lin))
            I_m_max_vals_bv.append(I_m_max_bv)
            I_m_max_vals_lin.append(I_m_max_lin)
            I_ds_bv = h / p.L * np.sum(I_m_bv)
            I_ds_lin = h / p.L * np.sum(I_m_lin)
            I_ds_vals_bv.append(I_ds_bv)
            I_ds_vals_lin.append(I_ds_lin)
            i_p_max_bv = np.max(port_current_density_bv)
            i_p_max_lin = np.max(port_current_density_lin)
            i_p_max_vals_bv.append(i_p_max_bv)
            i_p_max_vals_lin.append(i_p_max_lin)

            fig, ax = plt.subplots()
            ax.plot(0.5*p.N_s * y[:-2]/p.L, I_m_lin[1:], label="Linear")
            ax.plot(0.5*p.N_s * y[:-2]/p.L, I_m_bv[1:], label="Butler-Volmer")
            ax.set_xlim([0, 0.5 * p.N_s])
            ax.set_ylim([-2.5, 0])
            ax.set_xlabel("Cell number")
            ax.set_ylabel("Manifold current [A]")
            ax.grid(color="cyan", linewidth=0.1)
            ax.set_box_aspect(1)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "manifold", f"{var_value}.eps"))
            plt.close()

        fig, ax = plt.subplots()
        ax.semilogx(variables, i_p_max_vals_bv, 'r-.', label="Butler-Volmer")
        ax.semilogx(variables, i_p_max_vals_lin, 'r', label="Linear")
        ax2 = ax.twinx()
        ax2.semilogx(variables, np.abs(I_ds_vals_bv), 'b-.', label="Butler-Volmer")
        ax2.semilogx(variables, np.abs(I_ds_vals_lin), 'b', label="Linear")
        ax.set_ylim([400, 1400])
        ax2.set_ylim([1.5, 2.75])
        ax.set_ylabel(r"Maximum port current density [A/m$^2$]")
        ax2.set_ylabel(r"Manifold current [A]")
        ax.set_xlabel(r"$\omega$ [m$^{-1}$]")
        ax.set_box_aspect(1)
        ax.legend(loc="upper left")
        ax.spines["left"].set_color("red")
        ax.yaxis.label.set_color("red")
        ax.tick_params(colors="red", axis="y")
        ax2.spines["right"].set_color("blue")
        ax2.yaxis.label.set_color("blue")
        ax2.tick_params(colors="blue", axis="y")
        ax.grid(color="cyan", linewidth=0.1)
        ax2.yaxis.set_ticks([1.5, 1.75, 2.0, 2.25, 2.5, 2.75])
        ax.set_xlim([np.min(variables), np.max(variables)])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
        plt.savefig(os.path.join(results_dir, "I_manifold.eps"))
        plt.close()

    variables = vary_L_p()
    if args.vary == 'L_p':
        for L_p in variables:
            p = ShuntCurrentsParameters(L_p=L_p)
            h = p.N_s * p.d_p / 2 / N
            y = np.zeros((N+1, 1))
            for idx in range(N):
                y[idx] = idx * h
            eta_s0 = 1e-8 * np.ones((N+1, 1))
            u_lin, u_bv = solve_loop(N, h, p, eta_s0, tol=tol, max_its=max_its)

            if args.vary == "L_p":
                var_value = f"{p.L_p:.3f}"
            elif args.vary == "N_s":
                var_value = p.N_s
            elif args.vary == "w":
                var_value = p.omega
            else:
                raise ValueError("Unknown study type")
            
            fig, ax = plt.subplots()
            ax.plot(0.5*p.N_s * y[:-1]/p.L, u_lin[:-1], 'b', label="Linear")
            ax.plot(0.5*p.N_s * y[:-1]/p.L, u_bv[:-1], 'r-.', label="Butler-Volmer")
            ax.plot([0, 0.5 * p.N_s], [0, 50], linestyle='--', color='black', label="Electrode potential")
            ax.set_xlim([0, 0.5 * p.N_s])
            ax.set_ylim([0, 50])
            ax.set_xlabel("Cell number")
            ax.set_ylabel("Potential [V]")
            ax.grid(color="cyan", linewidth=0.1)
            ax.set_box_aspect(1)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "potential", f"{var_value}.eps"))
            plt.close()

            fig, ax = plt.subplots()
            port_current_density_bv = i_port_approx(u_bv, p)
            port_current_density_lin = i_port_approx(u_lin, p)
            ax.plot(0.5*p.N_s * y[:-1]/p.L, port_current_density_bv, 'r-.', label="Butler-Volmer")
            ax.plot(0.5*p.N_s * y[:-1]/p.L, port_current_density_lin, 'b', label="Linear")
            ax.grid(color="cyan", linewidth=0.1)
            ax.set_xlim([0, 0.5*p.N_s])
            ax.set_ylim([0, 1.01 * np.max(port_current_density_bv)])
            ax.set_box_aspect(1)
            ax.set_ylabel(r"$i_p$ [A/m$^2$]")
            ax.set_xlabel("Cell number")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "i_port", f"{var_value}.eps"))
            plt.close()
            I_m_bv = I_manifold(u_bv, p)
            I_m_lin = I_manifold(u_lin, p)
            I_m_max_bv = np.max(np.abs(I_m_bv))
            I_m_max_lin = np.max(np.abs(I_m_lin))
            I_m_max_vals_bv.append(I_m_max_bv)
            I_m_max_vals_lin.append(I_m_max_lin)
            I_ds_lin = h / p.L * np.sum(I_m_lin)
            I_ds_bv = h / p.L * np.sum(I_m_bv)
            I_ds_vals_bv.append(I_ds_bv)
            I_ds_vals_lin.append(I_ds_lin)
            i_p_max_bv = np.max(port_current_density_bv)
            i_p_max_lin = np.max(port_current_density_lin)
            i_p_max_vals_bv.append(i_p_max_bv)
            i_p_max_vals_lin.append(i_p_max_lin)

            fig, ax = plt.subplots()
            ax.plot(0.5*p.N_s * y[:-2]/p.L, I_m_lin[1:], 'b', label="Linear")
            ax.plot(0.5*p.N_s * y[:-2]/p.L, I_m_bv[1:], 'r-.', label="Butler-Volmer")
            ax.set_xlim([0, 0.5 * p.N_s])
            ax.set_ylim([-2.5, 0])
            ax.set_xlabel("Cell number")
            ax.set_ylabel("Manifold current [A]")
            ax.grid(color="cyan", linewidth=0.1)
            ax.set_box_aspect(1)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "manifold", f"{var_value}.eps"))
            plt.close()

        fig, ax2 = plt.subplots()
        ax = ax2.twinx()
        ax.plot(variables, i_p_max_vals_bv, 'r-.', label=r"Butler-Volmer")
        ax.plot(variables, i_p_max_vals_lin, 'r', label=r"Linear")
        ax2.plot(variables, np.abs(I_ds_vals_bv), 'b-.', label=r"Butler-Volmer")
        ax2.plot(variables, np.abs(I_ds_vals_lin), 'b', label=r"Linear")
        ax.set_ylim([500, 3000])
        ax2.set_ylim([1.0, 3.0])
        ax.set_ylabel(r"Maximum port current density [A/m$^2$]")
        ax2.set_ylabel(r"Manifold current [A]")
        ax.set_xlabel(r"$L_p$ [m]")
        ax.set_box_aspect(1)
        ax.legend(loc="upper left")
        ax.spines["right"].set_color("red")
        ax.yaxis.label.set_color("red")
        ax.tick_params(colors="red", axis="y")
        ax2.spines["left"].set_color("blue")
        ax2.yaxis.label.set_color("blue")
        ax2.tick_params(colors="blue", axis="y")
        ax.grid(color="cyan", linewidth=0.1)
        ax2.yaxis.set_ticks([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
        ax.set_xlim([np.min(variables), np.max(variables)])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
        plt.savefig(os.path.join(results_dir, "I_manifold.eps"))
        plt.close()

    variables = vary_N_s()
    if args.vary == 'N_s':
        for N_s in variables:
            p = ShuntCurrentsParameters(N_s=N_s)
            print(f"Omega: {p.omega}")
            h = p.N_s * p.d_p / 2 / N
            y = np.zeros((N+1, 1))
            for idx in range(N):
                y[idx] = idx * h
            eta_s0 = 1e-8 * np.ones((N+1, 1))
            u_lin, u_bv = solve_loop(N, h, p, eta_s0, tol=tol, max_its=max_its)

            if args.vary == "L_p":
                var_value = p.L_p
            elif args.vary == "N_s":
                var_value = p.N_s
            elif args.vary == "w":
                var_value = p.omega
            else:
                raise ValueError("Unknown study type")
            
            fig, ax = plt.subplots()
            ax.plot(0.5*p.N_s * y[:-1]/p.L, u_lin[:-1], label="Linear")
            ax.plot(0.5*p.N_s * y[:-1]/p.L, u_bv[:-1], label="Butler-Volmer")
            ax.plot([0, 0.5 * p.N_s], [0, 50], linestyle='--', color='cyan', label="Electrode potential")
            ax.set_xlim([0, 0.5 * p.N_s])
            ax.set_ylim([0, 50])
            ax.set_xlabel("Cell number")
            ax.set_ylabel("Potential [V]")
            ax.grid()
            ax.set_box_aspect(1)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "potential", f"{var_value}.eps"))
            plt.close()

            fig, ax = plt.subplots()
            port_current_density_lin = i_port_approx(u_lin, p)
            port_current_density_bv = i_port_approx(u_bv, p)
            ax.plot(0.5*p.N_s * y[:-1]/p.L, port_current_density_lin, label="Linear")
            ax.plot(0.5*p.N_s * y[:-1]/p.L, port_current_density_bv, label="Butler-Volmer")
            ax.legend()
            ax.grid(color='cyan')
            ax.set_xlim([0, 0.5*p.N_s])
            ax.set_ylim([0, 1.01 * np.max(port_current_density_bv)])
            ax.set_box_aspect(1)
            ax.set_ylabel(r"$i_p$ [A/m$^2$]")
            ax.set_xlabel("Cell number")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "i_port", f"{var_value}.eps"))
            plt.close()
            I_m_bv = I_manifold(u_bv, p)
            I_m_lin = I_manifold(u_lin, p)
            I_m_max_bv = np.max(np.abs(I_m_bv))
            I_m_max_lin = np.max(np.abs(I_m_lin))
            I_m_max_vals_bv.append(I_m_max_bv)
            I_m_max_vals_lin.append(I_m_max_lin)
            I_ds_lin = h / p.L * np.sum(I_m_lin)
            I_ds_bv = h / p.L * np.sum(I_m_bv)
            I_ds_vals_bv.append(I_ds_bv)
            I_ds_vals_lin.append(I_ds_lin)
            i_p_max_bv = np.max(port_current_density_bv)
            i_p_max_lin = np.max(port_current_density_lin)
            i_p_max_vals_bv.append(i_p_max_bv)
            i_p_max_vals_lin.append(i_p_max_lin)

            fig, ax = plt.subplots()
            ax.plot(0.5*p.N_s * y[:-2]/p.L, I_m_lin[1:], label="Linear")
            ax.plot(0.5*p.N_s * y[:-2]/p.L, I_m_bv[1:], label="Butler-Volmer")
            ax.set_xlim([0, 0.5 * p.N_s])
            ax.set_ylim([-2.5, 0])
            ax.set_xlabel("Cell number")
            ax.set_ylabel("Manifold current [A]")
            ax.grid()
            ax.set_box_aspect(1)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "manifold", f"{var_value}.eps"))
            plt.close()

        fig, ax = plt.subplots()
        ax.plot(variables, i_p_max_vals_bv, 'r-.', label=r"$i_{p,\mathrm{max}}$")
        ax.plot(variables, i_p_max_vals_lin, 'b', label=r"$i_{p,\mathrm{max}}$")
        ax2 = ax.twinx()
        ax2.plot(variables, np.abs(I_ds_vals_bv), 'r-.', label=r"$I_{\mathrm{ds}}$")
        ax2.plot(variables, np.abs(I_ds_vals_lin), 'b', label=r"$I_{\mathrm{ds}}$")
        ax.set_ylim([0.99 * np.min(i_p_max_vals_bv), 1.01 * np.max(i_p_max_vals_bv)])
        # ax2.set_ylim([1.5, 2.75])
        ax.set_ylabel(r"Maximum port current density [A/m$^2$]")
        ax2.set_ylabel(r"Manifold current [A]")
        ax.set_xlabel("Cell number")
        ax.set_box_aspect(1)
        ax.legend(loc="upper left")
        ax.spines["right"].set_color("red")
        ax.yaxis.label.set_color("red")
        ax.tick_params(colors="red", axis="y")
        ax2.spines["left"].set_color("blue")
        ax2.yaxis.label.set_color("blue")
        ax2.tick_params(colors="blue", axis="y")
        ax.grid(color="cyan", linewidth=0.1)
        ax2.yaxis.set_ticks([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
        ax.set_xlim([np.min(variables), np.max(variables)])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
        plt.savefig(os.path.join(results_dir, "I_manifold.eps"))
        plt.close()
