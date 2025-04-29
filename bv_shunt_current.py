#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import matspy

import solvers

R = 8.314
T = 298
F = 96485


def i_p(eta, p):
    return np.sqrt(2 * p.a * p.i0 * p.kappa * R * T/(p.a_a * p.a_c * F)) *\
        np.sqrt(p.a_c * np.exp(p.a_a * F * eta/(R * T)) +\
                p.a_a * np.exp(-p.a_c * F * eta/(R * T)) - p.a_a - p.a_c)


def i_p_prime(eta, p):
    return np.sqrt(2 * p.a * p.i0 * p.kappa * R * T/(p.a_a * p.a_c * F)) * (p.a_a * p.a_c * F / (R * T) ) * (np.exp(p.a_a * F * eta/(R * T)) -\
                np.exp(-p.a_c * F * eta/(R * T)))/\
            np.sqrt(p.a_c * np.exp(p.a_a * F * eta/(R * T)) + p.a_a * np.exp(-p.a_c * F * eta/(R * T)) - p.a_a - p.a_c)


def source(y, eta_s0, p):
    return (p.V_cell / p.d_p * y - eta_s0 + i_p(eta_s0, p)/i_p_prime(eta_s0, p))


def lambda_squared(eta_s0, p):
    return p.H_p / (p.kappa * p.A_m) * i_p_prime(eta_s0, p) / (i_p_prime(eta_s0, p) * p.R_p + 1)


def solve_for_manifold_potential(eta_s0, p, N, h):
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
            l2 = lambda_squared(eta_s0[idx], p)
        if idx < N - 1:
            b[idx] = -l2 * source((idx+1) * h, eta_s0[idx], p)
        else:
            print(idx)
        if idx == 0:  # y = h
            A[idx, idx+1] = 1/h**2
            A[idx, idx] = -2/h**2 - l2
        # elif idx == N-2:  # y = L
        #     A[idx, idx+1] = 1 * p.kappa / (2*h)
        #     A[idx, idx-1] = -1 * p.kappa / (2*h)
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
    comm = MPI.COMM_WORLD
    w = 1e2
    a = 100
    kappa = 4
    a_a = 0.5
    a_c = 0.5
    p = solvers.ShuntCurrentsSolver(comm=comm, a=a, kappa=kappa, a_a=a_a, a_c=a_c, i0=kappa*R*T*w**2/(F*a*(a_a+a_c)))
    N = 500
    h = p.N_s * p.d_p / 2 / N
    y = np.zeros((N+1, 1))
    for idx in range(N):
        y[idx] = idx * h
    eta_s0 = 1e-8 * np.ones((N+1, 1))
    A, b, u = solve_for_manifold_potential(eta_s0, p, N+1, h)
    # matspy.spy(A)
    # u = np.linalg.solve(A, b)
    fig, ax = plt.subplots()
    ax.plot(y[:-1], u[:-1])
    ax.plot([0, p.L], [0, 50], linestyle='--', color='cyan')
    ax.set_xlim([0, p.L])
    ax.set_ylim([0, 50])
    ax.grid()
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.show()
