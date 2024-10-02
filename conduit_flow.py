#!/usr/bin/env python3
import argparse
import json
import os
import gmsh
import matplotlib.pyplot as plt
import numpy as np
import ufl

from basix.ufl import element
from collections import defaultdict

from dolfinx import cpp, default_scalar_type, fem, io, mesh
from dolfinx.fem import petsc
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import gmshio, VTXWriter
from dolfinx.nls import petsc as petsc_nls
from mpi4py import MPI
from petsc4py import PETSc
from ufl import grad, inner

import commons, constants, utils

dtype = PETSc.ScalarType

class Labels:
    def __init__(self):
        pass

    @property
    def domain(self):
        return 1

    @property
    def inlet(self):
        return 1

    @property
    def outlet(self):
        return 2

    @property
    def inlet_outlet_separation(self):
        return 3

    @property
    def left(self):
        return 4

    @property
    def right(self):
        return 5

    @property
    def top(self):
        return 6

    @property
    def insulated(self):
        return 7


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='secondary current distribution')
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--k", help="permeability", nargs='?', const=1, default=1.0, type=float)
    parser.add_argument("--mu", help="viscosity", nargs='?', const=1, default=1.0, type=float)
    parser.add_argument("--p_in", help="inlet gauge pressure", nargs='?', const=1, default=1.0, type=float)
    parser.add_argument("--p_out", help="outlet gauge pressure", nargs='?', const=1, default=0, type=float)
    parser.add_argument("--Lc", help="characteristic length", nargs='?', const=1, default=1.0, type=float)
    parser.add_argument("--h_over_L", help="aspect ratio", nargs='?', const=1, default=0.1, type=float)
    parser.add_argument("--w_over_L", help="aspect ratio of inlet/outlet", nargs='?', const=1, default=0.1, type=float)
    args = parser.parse_args()
    mesh_folder = os.path.join("output", "conduit_flow")
    workdir = os.path.join(args.mesh_folder, str(args.Lc), str(args.h_over_L), str(args.w_over_L))
    utils.make_dir_if_missing(workdir)
    output_meshfile_path = os.path.join(workdir, "mesh.msh")
    markers = Labels()
    comm = MPI.COMM_WORLD

    output_current_path = os.path.join(workdir, 'current.bp')
    output_potential_path = os.path.join(workdir, 'potential.bp')
    simulation_metafile = os.path.join(workdir, 'simulation.json')

    print("Loading mesh..")
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = gmshio.read_from_msh(output_meshfile_path, comm, partitioner=partitioner)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)

    inlet_boundary = ft.find(markers.inlet)
    outlet_boundary = ft.find(markers.outlet)
    print("done\n")

    # Dirichlet BCs
    V = fem.functionspace(domain, ("CG", 2))
    
    n = ufl.FacetNormal(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    dx = ufl.Measure("dx", domain=domain, subdomain_data=ct)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Constant(domain, dtype(0.0))
    g = fem.Constant(domain, dtype(0.0))

    inlet_dofs = fem.locate_dofs_topological(V, fdim, inlet_boundary)
    outlet_dofs = fem.locate_dofs_topological(V, fdim, outlet_boundary)
    left_bc = fem.dirichletbc(dtype(args.p_in), inlet_dofs, V)
    right_bc = fem.dirichletbc(dtype(args.p_out), outlet_dofs, V)

    a_vv = inner(grad(u), grad(v)) * dx
    L_v = inner(f, v) * dx + inner(g, v) * ds(markers.insulated)
    print(f'Solving problem..')

    problem = petsc.LinearProblem(a_vv, L_v, bcs=[left_bc, right_bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    with VTXWriter(comm, output_potential_path, [uh], engine="BP5") as vtx:
        vtx.write(0.0)

    print("Post-process calculations")
    W = fem.functionspace(domain, ("CG", 2, (3,)))
    q_expr = fem.Expression(-(args.k / args.mu) * ufl.grad(uh), W.element.interpolation_points())
    q_h = fem.Function(W, name='current_density')
    q_h.interpolate(q_expr)
    norm_factor = (args.mu/args.k)*(args.Lc/np.abs(args.p_in - args.p_out))
    deltaP = np.abs(args.p_in - args.p_out)
    w = args.w_over_L * args.Lc
    Q_in =  np.abs(comm.allreduce(fem.assemble_scalar(fem.form(inner(q_h, n) * ds(markers.inlet))), op=MPI.SUM))
    Q_out =  np.abs(comm.allreduce(fem.assemble_scalar(fem.form(inner(q_h, n) * ds(markers.outlet))), op=MPI.SUM))
    r_tilde_in = (args.k/args.mu) / (Q_in * args.Lc * deltaP)
    r_tilde_out = (args.k/args.mu) / (Q_out * args.Lc * deltaP)
    simulation_metadata = {
        "Q in": Q_in,
        "Q out": Q_out,
        "rtilde in": r_tilde_in,
        "rtilde out": r_tilde_out,
        "h/L": args.h_over_L,
        "w/L": args.w_over_L,
        "Lc": args.Lc,
        "k": args.k,
        "mu": args.mu,
        "p in": args.p_in,
        "p out": args.p_out,
        "s/w": (1 - 2 * args.w_over_L) / args.w_over_L,
        "h/w": args.h_over_L / args.w_over_L,
    }

    with VTXWriter(comm, output_current_path, [q_h], engine="BP5") as vtx:
        vtx.write(0.0)

    if comm.rank == 0:
        utils.print_dict(simulation_metadata, padding=50)
        with open(simulation_metafile, "w", encoding='utf-8') as f:
            json.dump(simulation_metadata, f, ensure_ascii=False, indent=4)

