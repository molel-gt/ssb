#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os
import resource
import sys
import time
import timeit

os.environ["XDG_CACHE_HOME"] = os.path.join(os.getcwd(), ".cache/fenics", str(hash(tuple(sys.argv))))
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


a = 100 # [1/m]
i0 = 10 # [A/m2]
alpha_a = 0.5
alpha_c = 0.5
T = 298 # [K]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(desc="Estimation of shunt currents")
    parser.add_argument("--N_s", help="Number of cells in bipolar stack", nargs='?', const=1, default=100, type=int)
    parser.add_argument("--d_p", help="Cell pitch, distance between cells in bipolar configuration", nargs='?', const=1, default=100, type=int)
    parser.add_argument("--V_cell", help="Potential of individual cell at open-circuit potential", nargs='?', const=1, default=1.0, type=float)
    parser.add_argument("--kappa", help="Conductivity of the electrolyte", nargs='?', const=1, default=4, type=float)
    parser.add_argument("--H_p", help="Area per unit height for flow in the port", nargs='?', const=1, default=0.03, type=float)
    parser.add_argument("--A_m", help="Area for current flow in the manifold", nargs='?', const=1, default=0.006, type=float)
    parser.add_argument("--L_p", help="Length of the port", nargs='?', const=1, default=0.02, type=float)
    parser.add_argument("--w", help="kinetic parameter for linear kinetics", nargs='?', const=1, default=99, type=float)
