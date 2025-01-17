#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import resource
import time
import timeit

import basix
import dolfinx

import gmsh
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.special as sp
import ufl

from dolfinx import cpp, default_real_type, fem, io, mesh, log
from dolfinx.graph import partitioner_kahip
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.nls import petsc as petsc_nls
from matplotlib import rc
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
from ufl import dot, grad, inner

# import plot_opts
# plt.rcParams.update(plot_opts.params)

left = 1
right = 2
sides = 3
insulated = 4
left_surfs = []
right_surfs = []
side_surfs = []


def create_mesh(msh_output_path, resolution=0.035):
    gmsh.initialize()
    gmsh.model.add('mesh')
    gmsh.option.setNumber("Mesh.MeshSizeMax", resolution)
    box = gmsh.model.occ.addBox(-0.5, -0.5, 0, 0.5, 0.5, 1)
    gmsh.model.occ.synchronize()
    for surf in gmsh.model.getEntities(2):
        com = gmsh.model.occ.getCenterOfMass(*surf)
        if np.isclose(com[2], 0):
            left_surfs.append(surf[1])
        elif np.isclose(com[2], 1):
            right_surfs.append(surf[1])
        else:
            side_surfs.append(surf[1])
    gmsh.model.addPhysicalGroup(2, left_surfs, left, "left")
    gmsh.model.addPhysicalGroup(2, right_surfs, right, "right")
    gmsh.model.addPhysicalGroup(2, side_surfs, sides, "sides")
    gmsh.model.addPhysicalGroup(2, right_surfs + side_surfs, insulated, "insulated")

    gmsh.model.addPhysicalGroup(3, [gmsh.model.getEntities(3)[0][1]], 1, "domain")
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(msh_output_path)
    gmsh.finalize()


class Precon(PETSc.PC):
    def setUp(self):
        pass

    def apply(self):
        pass

class MatrixFreePC(object):
    def setUp(self, pc):
        B, P = pc.getOperators()
        # extract the MatrixFreeB object from B
        ctx = B.getPythonContext()
        self.A = ctx.A
        self.u = ctx.u
        self.v = ctx.v
        # Here we build the PC object that uses the concrete,
        # assembled matrix A.  We will use this to apply the action
        # of A^{-1}
        self.pc = PETSc.PC().create()
        self.pc.setOptionsPrefix("mf_")
        self.pc.setOperators(self.A)
        self.pc.setFromOptions()
        # Since u and v do not change, we can build the denominator
        # and the action of A^{-1} on u only once, in the setup
        # phase.
        tmp = self.A.createVecLeft()
        self.pc.apply(self.u, tmp)
        self._Ainvu = tmp
        self._denom = 1 + self.v.dot(self._Ainvu)

    def apply(self, pc, x, y):
        # y <- A^{-1}x
        self.pc.apply(x, y)
        # alpha <- (v^T A^{-1} x) / (1 + v^T A^{-1} u)
        alpha = self.v.dot(y) / self._denom
        # y <- y - alpha * A^{-1}u
        y.axpy(-alpha, self._Ainvu)


if __name__ == '__main__':
    output_meshfile = "mesh.msh"
    # create_mesh(output_meshfile)

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    partitioner = mesh.create_cell_partitioner(partitioner_kahip(), ghost_mode=mesh.GhostMode.shared_facet)
    domain, ct, ft = io.gmshio.read_from_msh(output_meshfile, comm, partitioner=partitioner)[:3]
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)
    domain.topology.create_connectivity(tdim, tdim)
    domain.topology.create_connectivity(fdim, fdim)

    dx = ufl.Measure('dx', domain=domain, subdomain_data=ct, subdomain_id=1)
    ds = ufl.Measure('ds', domain=domain, subdomain_data=ft)

    g = fem.Constant(domain, -50.0)
    g0 = fem.Constant(domain, 0.0)

    n = ufl.FacetNormal(domain)
    dt = fem.Constant(domain, 1e-4)
    el = basix.ufl.element(basix.ElementFamily.P, basix.CellType.tetrahedron, 4, basix.LagrangeVariant.gll_isaac, dtype=default_real_type)
    el = ("CG", 4)
    VC = fem.functionspace(domain, el)

    VC_dofmap = VC.dofmap
    VC_map = VC.dofmap.index_map

    n_dofs = VC_map.size_global*VC.dofmap.index_map_bs

    PETSc.Sys.Print(f"#DoFs: {n_dofs}")

    c, q = fem.Function(VC), ufl.TestFunction(VC)
    c0 = fem.Function(VC)

    c0.interpolate(lambda x: x[2] - x[2] + 0.75)
    c.interpolate(lambda x: 0.75 * (1 - np.exp(-x[2])))

    k1 = ufl.inner(ufl.grad(c0), ufl.grad(q))
    k2 = ufl.inner(ufl.grad(c0 + dt/2 * k1), ufl.grad(q))
    k3 = ufl.inner(ufl.grad(c0 + dt/3 * k2), ufl.grad(q))
    k4 = ufl.inner(ufl.grad(c0 + dt * k3), ufl.grad(q))
    # F = (c - c0)/dt * q * dx + inner(ufl.grad(c), ufl.grad(q)) * dx
    f_rk4 = 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    F = (c - c0)/dt * q * dx + f_rk4 * dx
    F += -g * q * ds(left) + g0 * q * ds(insulated)
    max_time = 1 * dt.value
    t = 0

    while t < max_time:
        t += dt.value
        problem = fem.petsc.NonlinearProblem(F, c, bcs=[])
        solver = petsc_nls.NewtonSolver(comm, problem)
        solver.convergence_criterion = "residual"
        solver.maximum_iterations = 100
        # solver.atol = np.finfo(float).eps
        solver.rtol = 1e-8 #np.finfo(float).eps * 10

        ksp = solver.krylov_solver
        # pc = ksp.getPC()
        # pc.setType(pc.Type.PYTHON)
        # mpc = MatrixFreePC()
        # pc.setPythonContext(mpc)
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "cg"
        opts[f"{option_prefix}pc_type"] = "sor"
        opts['log_view'] = None
        opts[f'{option_prefix}ksp_monitor_singular_value'] = None
        # gamg = {
        #     "pc_gamg_type": 'agg',
        #     "pc_gamg_threshold": 0.05,
        #     "pc_gamg_repartition": True,
        #     "pc_gamg_aggressive_coarsening": 4,
        #     "pc_gamg_aggressive_square_graph": 1,
        #     "pc_gamg_agg_nsmooths": 0,
        #     "pc_gamg_coarse_eq_limit": 10000,
        #     "pc_gamg_parallel_coarse_grid_solver": True,
        #     "pc_gamg_eigenvalues": [1e-4, 50],
        #     "pc_gamg_use_sa_esteig": True,
        #     }
        # for kopt, vopt in gamg.items():
        #     opts[f"{option_prefix}{kopt}"] = vopt
        ksp.setFromOptions()
        start = time.time()
        PETSc.Log().begin()
        n_iters, converged = solver.solve(c)
        end = time.time()
        print(f"{n}, solve time: {end-start}")

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
        bb_trees = bb_tree(domain, domain.topology.dim)
        # Find cells whose bounding-box collide with the the points
        cell_candidates = compute_collisions_points(bb_trees, points.T)
        # Choose one of the cells that contains the point
        colliding_cells = compute_colliding_cells(domain, cell_candidates, points.T)

        for i in range(n_points):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(points.T[i])
                cells.append(colliding_cells.links(i)[0])

        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        c_values_mid = c.eval(points_on_proc, cells)
        if np.all(c_values_mid.shape):
            try:
                c_plot_vals = np.hstack((points_on_proc, c_values_mid))
            except ValueError:
                c_plot_vals = np.empty((0, 4))
        else:
            c_plot_vals = np.empty((0, 4))

        if comm_rank != 0:
            req = comm.send(c_plot_vals, dest=0, tag=11)

        if comm_rank == 0:
            all_c_vals = c_plot_vals
            for rank in range(1, comm_size):
                addtnl_c = comm.recv(source=rank, tag=11)
                all_c_vals = np.vstack((all_c_vals, addtnl_c))

            c_vals = all_c_vals[all_c_vals[:, 2].argsort()]

            fig, ax = plt.subplots()
            ax.plot(c_vals[:, 2], c_vals[:, 3], 'k', label=r'0.5$L_x$,0.5$L_y$', linewidth=1)
            ax.grid(True)
            ax.legend()
            # ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_box_aspect(1)
            ax.set_ylabel(r'$\hat{c}$', rotation=90, labelpad=0, fontsize='xx-large')
            ax.set_xlabel(r'$\hat{x}$')
            plt.tight_layout()
            plt.savefig("concentration.eps")
            # plt.show()
