#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os

import dolfinx
import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
import pyvista
import pyvista as pv
import pyvistaqt as pvqt
import ufl
import warnings

from dolfinx import cpp, default_scalar_type, fem, io, la, mesh, nls, plot
from dolfinx.fem import petsc
from dolfinx.io import gmshio, VTXWriter
from dolfinx.nls import petsc as petsc_nls
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from IPython.display import Image

from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 dot, div, dx, ds, dS, grad, inner, grad, avg, jump)

import commons, configs, geometry, utils

warnings.simplefilter('ignore')


class SNESNonlinearProblem:
    def __init__(self, F, u):
        V = u.function_space
        du = ufl.TrialFunction(V)
        self.L = fem.form(F)
        self.a = fem.form(ufl.derivative(F, u, du))
        # self.bc = bc
        self._F, self._J = None, None
        self.u = u
        self.J_mat_dolfinx = petsc.create_matrix(self.a)
        self.F_vec_dolfinx = petsc.create_vector(self.L)

    def F(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        self.F_vec_dolfinx.set(0.)
        with F.localForm() as f_local:
            f_local.set(0.0)
        petsc.assemble_vector(self.F_vec_dolfinx, self.L)
        petsc.apply_lifting(self.F_vec_dolfinx, [self.a], bcs=[[]], x0=[x], scale=-1.0)
        self.F_vec_dolfinx.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(self.F_vec_dolfinx, [], x, -1.0)
        F.getArray()[:] = self.F_vec_dolfinx.getArray()[:]

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        self.J_mat_dolfinx.zeroEntries()

        fem.petsc.assemble_matrix(self.J_mat_dolfinx, self.a, bcs=[])
        self.J_mat_dolfinx.assemble()
        ai, aj, av = self.J_mat_dolfinx.getValuesCSR()
        J.setPreallocationNNZ(np.diff(ai))
        J.setValuesCSR(ai, aj, av)
        J.assemble()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Current Collector.')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="lithium_metal_3d_cc_2d")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1.0, type=float)
    parser.add_argument("--Wa", help="Wagna number: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e3, type=float)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='MICRON_TO_METER', type=str)

    args = parser.parse_args()
    Wa = args.Wa
    Wa_n = Wa
    Wa_p = Wa
    comm = MPI.COMM_WORLD
    encoding = io.XDMFFile.Encoding.HDF5
    micron = 1e-6
    resolution = micron
    kappa_elec = 0.1
    kappa_pos_am = 0.2
    name_of_study = args.name_of_study
    dimensions = args.dimensions
    LX, LY, LZ = [float(vv) * micron for vv in dimensions.split("-")]
    meshdir = args.mesh_folder
    workdir = os.path.join(meshdir, str(Wa_n) + "-" + str(Wa_p))
    utils.make_dir_if_missing(meshdir)
    utils.make_dir_if_missing(workdir)
    output_meshfile = os.path.join(meshdir, 'mesh.msh')
    potential_resultsfile = os.path.join(workdir, "potential.bp")
    concentration_resultsfile = os.path.join(workdir, "concentration.bp")
    current_resultsfile = os.path.join(workdir, "current.bp")
    simulation_metafile = os.path.join(workdir, "simulation.json")

    markers = commons.Markers()


    # ## Binary Electrolyte - Nernst-Planck Equation
    # 
    # We make the following assumptions
    # - no bulk flow
    # - diffusivity not a function of concentration
    # - binary electrolyte
    # 
    # $$\frac{\partial c}{\partial t} + \pmb{v}\cdot\nabla c - D\nabla^2c=0$$

    # ### [Electronic Conductivities](https://periodictable.com/Properties/A/ElectricalConductivity.an.html)
    # $$\sigma_{Li} = 1.1e7 \mathrm{\ S/m}$$
    # $$\sigma_{Cu} = 5.9e7 \mathrm{\ S/m}$$
    # $$\sigma_{Al} = 3.8e7 \mathrm{\ S/m}$$

    # #### Solid and Solution Parameters
    # 
    # From Chen 2020 *Journal of The Electrochemical Society, 2020 167 080534*, we can obtain parameters for use with our experiment. Here $c_e$ is concentration of $Li^+$ within the electrolyte in moles per cubic decimeter \[$\mathrm{mol}\cdot dm^{-3}$\].
    # 
    # Electronic conductivity of the electrolyte is given by:
    # $$\sigma_e = 0.1297c_e^3 - 2.51c_e^{1.5} + 3.329c_e$$
    # 
    # The diffusivity is given by:
    # $$D_e = 8.794 \cdot 10^{-11}c_e^2 - 3.972 \cdot 10^{-10}c_e + 4.862 \cdot 10^{-10}$$
    # 
    # The lithium ion transference number is given by:
    # $$t^+ = -0.1287c_e^3 + 0.4106c_e^2 - 0.4717c_e + 0.4492$$
    # 
    # Other parameters are given by

    # ### Read input geometry
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = gmshio.read_from_msh(output_meshfile, comm, partitioner=partitioner)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)

    ft_imap = domain.topology.index_map(fdim)
    num_facets = ft_imap.size_local + ft_imap.num_ghosts
    indices = np.arange(0, num_facets)
    values = np.zeros(indices.shape, dtype=np.intc)  # all facets are tagged with zero

    values[ft.indices] = ft.values
    ft = mesh.meshtags(domain, fdim, indices, values)
    ct = mesh.meshtags(domain, tdim, ct.indices, ct.values)
    dx = ufl.Measure("dx", domain=domain, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=ft)

    # full_mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(output_meshfile, comm, 0)

    # # Create submesh for pe
    # # domain, entity_map, vertex_map, geom_map = dolfinx.mesh.create_submesh(full_mesh, full_mesh.topology.dim, cell_tags.indices[(cell_tags.values == markers.electrolyte)])
    # domain, entity_map, vertex_map, geom_map = dolfinx.mesh.create_submesh(full_mesh, full_mesh.topology.dim, cell_tags.indices[np.logical_or(cell_tags.values == markers.electrolyte, cell_tags.values == markers.positive_am)])

    # # Transfer facet tags from parent mesh to submesh
    # tdim = full_mesh.topology.dim
    # fdim = tdim - 1
    # c_to_f = full_mesh.topology.connectivity(tdim, fdim)
    # f_map = full_mesh.topology.index_map(fdim)
    # all_facets = f_map.size_local + f_map.num_ghosts
    # all_values = np.zeros(all_facets, dtype=np.int32)
    # all_values[facet_tags.indices] = facet_tags.values

    # domain.topology.create_entities(fdim)
    # subf_map = domain.topology.index_map(fdim)
    # domain.topology.create_connectivity(tdim, fdim)
    # c_to_f_sub = domain.topology.connectivity(tdim, fdim)
    # num_sub_facets = subf_map.size_local + subf_map.num_ghosts
    # sub_values = np.empty(num_sub_facets, dtype=np.int32)
    # for i, entity in enumerate(entity_map):
    #     parent_facets = c_to_f.links(entity)
    #     child_facets = c_to_f_sub.links(i)
    #     for child, parent in zip(child_facets, parent_facets):
    #         sub_values[child] = all_values[parent]

    # ft = dolfinx.mesh.meshtags(domain, domain.topology.dim - 1, np.arange(
    #     num_sub_facets, dtype=np.int32), sub_values)
    # domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    # ft_imap = domain.topology.index_map(fdim)
    # num_facets = ft_imap.size_local + ft_imap.num_ghosts
    # indices = np.arange(0, num_facets)
    # values = np.zeros(indices.shape, dtype=np.intc)  # all facets are tagged with zero

    # values[ft.indices] = ft.values
    # ft = mesh.meshtags(domain, fdim, indices, values)

    # dx = ufl.Measure("dx", domain=domain)
    # ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    # dS = ufl.Measure("dS", domain=domain, subdomain_data=ft)


    # ### Function Spaces
    V = fem.FunctionSpace(domain, ("DG", 1))
    W = fem.functionspace(domain, ("CG", 1, (3,)))
    Q = fem.FunctionSpace(domain, ("DG", 0))
    u = fem.Function(V, name='potential')
    v = ufl.TestFunction(V)
    current_h = fem.Function(W, name='current_density')
    kappa = fem.Function(Q, name='conductivity')

    n = ufl.FacetNormal(domain)
    x = ufl.SpatialCoordinate(domain)
    h = ufl.CellDiameter(domain)
    h_avg = avg(h)

    cells_elec = ct.find(markers.electrolyte)
    kappa.x.array[cells_elec] = np.full_like(cells_elec, kappa_elec, dtype=default_scalar_type)

    cells_pos_am = ct.find(markers.positive_am)
    kappa.x.array[cells_pos_am] = np.full_like(cells_pos_am, kappa_pos_am, dtype=default_scalar_type)

    x = SpatialCoordinate(domain)

    f = fem.Constant(domain, PETSc.ScalarType(0))
    g = fem.Constant(domain, PETSc.ScalarType(0))

    voltage = 1.0
    u_left = fem.Function(V)
    with u_left.vector.localForm() as u0_loc:
        u0_loc.set(0)
    u_right = fem.Function(V)
    with u_right.vector.localForm() as u1_loc:
        u1_loc.set(voltage)


    faraday_const = fem.Constant(domain, PETSc.ScalarType(96485))
    R = fem.Constant(domain, PETSc.ScalarType(8.3145))
    T = fem.Constant(domain, PETSc.ScalarType(298))
    i0 = fem.Constant(domain, PETSc.ScalarType(kappa_elec * R * T / (Wa * faraday_const * LX)))
    i0_n = fem.Constant(domain, PETSc.ScalarType(kappa_elec * R * T / (Wa_n * faraday_const * LX)))
    i0_p = fem.Constant(domain, PETSc.ScalarType(kappa_elec * R * T / (Wa_p * faraday_const * LX)))
    Id = ufl.Identity(3)
    def ocv(sod, L=1, k=2):
        return 2.5 + (1/k) * np.log((L - sod) / sod)
    sod = 0.975

    U_n = 0
    U_vec = ufl.as_vector((0, 0, 0))
    # U_p = ufl.as_vector((ocv(sod), ocv(sod), ocv(sod)))
    U_p = ufl.as_vector((0, 0, 0))
    V_left = 0


    # ### Discontinuous Galerkin
    alpha = 10
    γ = 10

    F = kappa * inner(grad(u), grad(v)) * dx 
    F -= + f * v * dx 
    # F += - kappa * inner(grad(u), n) * v * ds
    F += - kappa * inner(grad(u), n) * v * ds(markers.negative_cc_v_negative_am)
    F += - kappa * inner(grad(u), n) * v * ds(markers.insulated_electrolyte)
    F += - kappa * inner(grad(u), n) * v * ds(markers.right)

    # Add DG/IP terms
    F += - inner(jump(kappa * u, n), avg(grad(v))) * dS(0)
    F += - inner(avg(kappa * grad(u)), jump(v, n)) * dS(0)
    F += alpha / h_avg * inner(jump(v, n), jump(kappa * u, n)) * dS(0)

    # left boundary - dirichlet
    # F += - kappa * (u - u_left) * inner(n, grad(v)) * ds(markers.left)
    # F += γ / h * (u - u_left) * v * ds(markers.left)

    # trial charge transfer
    # F += - inner(avg(grad(v)), jump(kappa, n) * avg(u) + avg(kappa) * (U_p - R * T / i0_p / faraday_const * (kappa * grad(u))('-'))) * dS(markers.electrolyte_v_positive_am)
    # F += - inner(jump(v, n), avg(grad(u))) * dS(markers.electrolyte_v_positive_am)
    # F += + alpha / h_avg * inner(jump(kappa, n) * avg(u) + avg(kappa) * (U_p - R * T / i0_p / faraday_const * (kappa * grad(u))('+')), jump(v, n)) * dS(markers.electrolyte_v_positive_am)

    # charge xfer internal boundary - neumann
    F += + inner(avg(grad(v)), jump(kappa, n) * avg(u) + avg(kappa) * (U_p - R * T / i0_p / faraday_const * (kappa * grad(u))('-'))) * dS(markers.electrolyte_v_positive_am)
    F += - alpha / h_avg * inner(jump(v, n), jump(kappa, n) * avg(u) + avg(kappa) * (U_p - (R * T / i0_p / faraday_const) * (kappa * grad(u))('-'))) * dS(markers.electrolyte_v_positive_am)
    F += + inner(avg(grad(v)), jump(kappa, n) * avg(u) + avg(kappa) * (U_p - R * T / i0_p / faraday_const * (kappa * grad(u))('+'))) * dS(markers.electrolyte_v_positive_am)
    F += - alpha / h_avg * inner(jump(v, n), jump(kappa, n) * avg(u) + avg(kappa) * (U_p - (R * T / i0_p / faraday_const) * (kappa * grad(u))('+'))) * dS(markers.electrolyte_v_positive_am)
    # F += inner(avg(kappa * grad(u)), jump(v, n)) * dS(markers.electrolyte_v_positive_am)

    # F += - inner(avg(grad(v)), U_p - R * T / i0 / faraday_const * (kappa * grad(u))('+')) * dS(markers.electrolyte_v_positive_am)
    # F += + alpha / h_avg * inner(jump(v, n), U_p - (R * T / i0_p / faraday_const) * (kappa * grad(u))('-')) * dS(markers.electrolyte_v_positive_am)
    # F += - inner(avg(grad(v)), U_p - R * T / i0 / faraday_const * (kappa * grad(u))('+')) * dS(markers.electrolyte_v_positive_am)
    # F += + alpha / h_avg * inner(jump(v, n), U_p - (R * T / i0_p / faraday_const) * (kappa * grad(u))('-')) * dS(markers.electrolyte_v_positive_am)

    # charge transfer terms
    # F += - dot(avg(grad(v)), (R * T / i0 / faraday_const) * (kappa * grad(u))('+') + U_p) * dS(markers.electrolyte_v_positive_am)
    # F += alpha / h_avg * dot(jump(v, n), (R * T / i0_p / faraday_const) * (kappa * grad(u))('+') + U_p) * dS(markers.electrolyte_v_positive_am)
    # F += - dot(avg(grad(v)), (R * T / i0 / faraday_const) * (kappa * grad(u))('-') + U_p) * dS(markers.electrolyte_v_positive_am)
    # F += alpha / h_avg * dot(jump(v, n), (R * T / i0_p / faraday_const) * (kappa * grad(u))('-') + U_p) * dS(markers.electrolyte_v_positive_am)

    # # charge xfer internal boundary - symmetry
    F += - inner(jump(kappa, n) * avg(u) + avg(kappa) * jump(u, n), avg(grad(v))) * dS(markers.electrolyte_v_positive_am)
    # # charge xfer internal boundary - coercivity
    F += + alpha / h_avg * inner(jump(kappa, n) * avg(u) + avg(kappa) * jump(u, n), jump(v, n)) * dS(markers.electrolyte_v_positive_am)


    # right boundary - dirichlet
    F += - kappa * (u - u_right) * inner(n, grad(v)) * ds(markers.right) 
    F += 1 / γ / h * (u - u_right) * v * ds(markers.right)

    # insulated boundary - neumann
    F += - γ * h * inner(inner(kappa * grad(u), n), inner(grad(v), n)) * ds(markers.insulated_electrolyte)
    F -= + γ * h * g * inner(grad(v), n) * ds(markers.insulated_electrolyte)

    # insulated boundary - neumann
    F += - γ * h * inner(inner(kappa * grad(u), n), inner(grad(v), n)) * ds(markers.insulated_positive_am)
    F -= + γ * h * g * inner(grad(v), n) * ds(markers.insulated_positive_am)

    # kinetics boundary - neumann
    F += - γ * h * inner(inner(kappa * grad(u), n), inner(grad(v), n)) * ds(markers.negative_cc_v_negative_am)
    F -= - γ * h * i0_n * faraday_const / R / T * (V_left - u - U_n) * inner(grad(v), n) * ds(markers.negative_cc_v_negative_am)

    # charge xfer external boundary - neumann
    # F += - γ * h * inner(inner(kappa * grad(u), n), inner(grad(v), n)) * ds(markers.electrolyte_v_positive_am)
    # F -= + γ * h * i0 * faraday_const / R / T  * (u - 0) * inner(grad(v), n) * ds(markers.electrolyte_v_positive_am)

    problem = petsc.NonlinearProblem(F, u)
    solver = petsc_nls.NewtonSolver(comm, problem)
    solver.convergence_criterion = "residual"
    solver.maximum_iterations = 25
    # solver.atol = 5e-12
    # solver.rtol = 1e-11

    ksp = solver.krylov_solver

    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    # opts[f"{option_prefix}ksp_type"] = "gmres"
    # opts[f"{option_prefix}pc_type"] = "gamg"

    ksp.setFromOptions()
    n_iters, converged = solver.solve(u)
    if converged:
        current_expr = fem.Expression(-kappa * grad(u), W.element.interpolation_points())
        current_h.interpolate(current_expr)
        print(f"Converged in {n_iters} iterations")

    # problem = SNESNonlinearProblem(F, u)
    # J_mat_dolfinx = petsc.create_matrix(problem.a)
    # F_vec_dolfinx = petsc.create_vector(problem.L)
    # J = PETSc.Mat().createAIJ(J_mat_dolfinx.getSizes())
    # # J.setLGMap(J_mat_dolfinx.getLGMap()[0],J_mat_dolfinx.getLGMap()[1]) # Only needed when assembling straight into J in problem.J
    # J.setPreallocationNNZ(30)
    # J.setUp()
    # b = J.createVecRight()
    # # b = la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
    # # J = petsc.create_matrix(problem.a)
    # snes = PETSc.SNES().create(comm)
    # snes.setTolerances(atol=1e-12, rtol=1.0e-11, max_it=10)
    # snes.getKSP().setType("preonly")
    # snes.getKSP().getPC().setType("lu")
    # snes.setFunction(problem.F, b)
    # snes.setJacobian(problem.J, J=J)

    # snes.setMonitor(lambda _, it, residual: print(it, residual))
    # snes.solve(None, u.vector)
    # if snes.getConvergedReason() > 0:
    #     current_expr = fem.Expression(-kappa * grad(u), W.element.interpolation_points())
    #     current_h.interpolate(current_expr)
    #     snes.destroy()
    #     b.destroy()
    #     J.destroy()


    # ### Continuous Galerkin
    # i_exchange = i0
    # kappa = fem.Constant(domain, PETSc.ScalarType(kappa_elec))
    # f = fem.Constant(domain, PETSc.ScalarType(0.0))
    # g = fem.Constant(domain, PETSc.ScalarType(0.0))

    # right_boundary = ft.find(markers.right)
    # right_bc = fem.dirichletbc(u_right, fem.locate_dofs_topological(V, 1, right_boundary))

    # F = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    # F += - ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ds(markers.insulated_electrolyte) 
    # F += + ufl.inner(i_exchange * faraday_const * (u - 0) / (R * T), v) * ds(markers.negative_cc_v_negative_am)

    # problem = petsc.NonlinearProblem(F, u, bcs=[right_bc])
    # solver = petsc_nls.NewtonSolver(comm, problem)
    # solver.convergence_criterion = "residual"
    # solver.maximum_iterations = 100
    # solver.atol = np.finfo(float).eps
    # solver.rtol = np.finfo(float).eps * 10

    # ksp = solver.krylov_solver
    # opts = PETSc.Options()
    # option_prefix = ksp.getOptionsPrefix()
    # opts[f"{option_prefix}ksp_type"] = "gmres"
    # opts[f"{option_prefix}pc_type"] = "hypre"
    # ksp.setFromOptions()
    # n_iters, converged = solver.solve(u)
    # if not converged:
    #     print(f"Solver did not converge in {n_iters} iterations")
    # else:
    #     print(f"Converged in {n_iters} iterations")
    # u.name = 'potential'
    # u.x.scatter_forward()

    with VTXWriter(comm, potential_resultsfile, [u], engine="BP4") as vtx:
        vtx.write(0.0)

    with VTXWriter(comm, current_resultsfile, [current_h], engine="BP4") as vtx:
        vtx.write(0.0)

    I_neg_charge_xfer = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(current_h, n) * ds(markers.negative_cc_v_negative_am))), op=MPI.SUM)
    I_pos_charge_xfer = domain.comm.allreduce(fem.assemble_scalar(fem.form(i0_p * faraday_const / R / T * ((u("+") - u("-") - U_p(0))) * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)
    I_right = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(current_h, n) * ds(markers.right))), op=MPI.SUM)
    I_insulated_elec = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(inner(current_h, n)) * ds(markers.insulated_electrolyte))), op=MPI.SUM)
    I_insulated_pos_am = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(inner(current_h, n)) * ds(markers.insulated_positive_am))), op=MPI.SUM)
    I_insulated = I_insulated_elec + I_insulated_pos_am
    area_left = domain.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * ds(markers.left))), op=MPI.SUM)
    area_neg_charge_xfer = domain.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * ds(markers.negative_cc_v_negative_am))), op=MPI.SUM)
    area_pos_charge_xfer = domain.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)
    area_right = domain.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * ds(markers.right))), op=MPI.SUM)
    i_sup_left = np.abs(I_neg_charge_xfer / area_neg_charge_xfer)
    i_sup = np.abs(I_right / area_right)
    eta = np.abs(i_sup_left) * (Wa  * LX / kappa_elec)
    eta_p = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs((u("+") - u("-") - U_p(0))) * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM) / area_pos_charge_xfer
    simulation_metadata = {
        "Negative Overpotential [V]": eta,
        "Positive Overpotential [V]": eta_p,
        "Voltage": voltage,
        "dimensions": dimensions,
        "i_sup [A/m2]": f"{np.abs(i_sup):.2e} [A/m2]",
        "Current at negative am - electrolyte boundary": f"{np.abs(I_neg_charge_xfer):.2e} A",
        "Current at electrolyte - positive am boundary": f"{np.abs(I_pos_charge_xfer):.2e} A",
        "Current at right boundary": f"{np.abs(I_right):.2e} A",
        "Current at insulated boundary": f"{I_insulated:.2e} A",
    }
    print(f"Voltage: {voltage} [V]")
    print(f"Negative Overpotential: {eta:.3e} [V]")
    print(f"Positive Overpotential: {eta_p:.3e} [V]")
    print(f"superficial current density @ left: {np.abs(i_sup_left):.2e} [A/m2]")
    print(f"superficial current density @ right: {np.abs(i_sup):.2e} [A/m2]")
    print(f"Current at negative_am - electrolyte boundary: {np.abs(I_neg_charge_xfer):.2e} A")
    print(f"Current at electrolyte - positive am boundary: {np.abs(I_pos_charge_xfer):.2e} A")
    print(f"Current at right boundary: {np.abs(I_right):.2e} A")
    print(f"Current at insulated boundary: {I_insulated:.2e} A")
    print(f"Float precision is {np.finfo(float).eps}")
    with open(simulation_metafile, "w", encoding='utf-8') as f:
        json.dump(simulation_metadata, f, ensure_ascii=False, indent=4)

    bb_trees = bb_tree(domain, domain.topology.dim)
    n_points = 100000
    tol = 1e-10
    points = np.zeros((3, n_points))
    fig, ax = plt.subplots()
    colors = ['red', 'blue', 'green']
    x = np.linspace(0 + tol, LX - tol, n_points)
    points[0] = x
    for idx, frac in enumerate([0.2, 0.5, 0.8]):    
        y = np.ones(n_points) * frac * LY  # position
        points[1] = y
        u_values = []
        cells = []
        points_on_proc = []
        # Find cells whose bounding-box collide with the the points
        cell_candidates = compute_collisions_points(bb_trees, points.T)
        # Choose one of the cells that contains the point
        colliding_cells = compute_colliding_cells(domain, cell_candidates, points.T)
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        u_values = u.eval(points_on_proc, cells)
        ax.plot((1/micron) * points_on_proc[:, 0], u_values, "-", linewidth=2, label=f'{frac}' + r' $L_y$', color=colors[idx])
    ax.set_xlim([0, LX/micron])
    ax.set_ylim([0, 1.1 * voltage])

    ax.set_ylabel(r'$\phi$ [V]', rotation=90, labelpad=0, fontsize='xx-large')
    ax.set_xlabel(r'$\mathrm{x}$ ' + '[$\mu$m]')
    ax.axhline(y=eta, linestyle='--', linewidth=0.5, label=r'$\eta$')
    ax.legend()
    ax.set_title(f'Wa = {Wa}')
    ax.minorticks_on();
    ax.tick_params(which="both", left=True, right=True, bottom=True, top=True, labelleft=True, labelright=False, labelbottom=True, labeltop=False);
    plt.tight_layout()
    plt.savefig(os.path.join(workdir, 'potential-dist-midline.png'))
    plt.show()

    bb_trees = bb_tree(domain, domain.topology.dim)
    points = np.zeros((3, n_points))
    fig, ax = plt.subplots()
    for idx, frac in enumerate([0.25, 0.5, 0.75]):
        if np.isclose(frac, 0.25, atol=1e-3) or np.isclose(frac, 0.75, atol=1e-3):
            x = np.linspace(10 * micron + tol, LX - tol, n_points)
        else:
            x = np.linspace(0 + tol, LX - tol, n_points)
        points[0] = x
        y = np.ones(n_points) * frac * LY
        points[1] = y
        u_values = []
        cells = []
        points_on_proc = []
        cell_candidates = compute_collisions_points(bb_trees, points.T)
        colliding_cells = compute_colliding_cells(domain, cell_candidates, points.T)
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        current_values = current_h.eval(points_on_proc, cells)
        ax.plot((1/micron) * points_on_proc[:, 0], np.linalg.norm(current_values, axis=1), "-", linewidth=1.5, color=colors[idx], label=f'{frac}'+r' $L_y$')
    ax.axhline(y=i_sup, color='black', linestyle='--', label=r'$i_{\mathrm{sup}}$')
    # ax.grid(True)
    ax.legend()
    ax.set_xlim([0, LX / micron])
    ax.set_ylabel(r'$\|{\mathbf{i}}\|$' + r' [Am$^{-2}$]', rotation=90, labelpad=0, fontsize='xx-large')
    ax.set_xlabel(r'$\mathrm{x}$ ' + r'[$\mu$m]')
    ax.set_title(f'Wa = {Wa}')
    ax.minorticks_on();
    ax.tick_params(which="both", left=True, right=True, bottom=True, top=True, labelleft=True, labelright=False, labelbottom=True, labeltop=False);
    plt.tight_layout()
    plt.savefig(os.path.join(workdir, 'current-dist-midline.png'))
    plt.show()

    bb_trees = bb_tree(domain, domain.topology.dim)
    points = np.zeros((3, n_points))
    fig, ax = plt.subplots()
    # fracs = [15e-6, 30e-6, 50e-6]
    fracs = [20e-6, 60e-6, 125e-6]
    for idx, frac in enumerate(fracs):
        y = np.linspace(0 + tol, LY - tol, n_points)
        points[1] = y
        x = np.ones(n_points) * frac
        points[0] = x
        u_values = []
        cells = []
        points_on_proc = []
        cell_candidates = compute_collisions_points(bb_trees, points.T)
        colliding_cells = compute_colliding_cells(domain, cell_candidates, points.T)
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        current_values = current_h.eval(points_on_proc, cells)
        ax.plot((1/micron) * points_on_proc[:, 1], current_values[:, 1], "-", linewidth=2, color=colors[idx], label=r' $\mathrm{x}$ = ' + f'{int(frac/micron)} ' + r'$\mu$m')
    ax.legend()
    # ax.set_xlim([0, 0.2 * LY / micron])
    ax.set_xlim([0, LY / micron])
    ax.set_ylabel(r'$i_y$ [Am$^{-2}$]', rotation=90, labelpad=0, fontsize='xx-large')
    ax.set_xlabel(r'$\mathrm{y}$ ' + r'[$\mu$m]')
    ax.set_title(f'Wa = {Wa}')
    ax.minorticks_on();
    ax.tick_params(which="both", left=True, right=True, bottom=True, top=True, labelleft=True, labelright=False, labelbottom=True, labeltop=False);
    plt.tight_layout()
    plt.savefig(os.path.join(workdir, 'current-dist-y-vertical.png'))
    plt.show()
