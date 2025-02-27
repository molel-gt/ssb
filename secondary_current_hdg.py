#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
import time

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import scifem
import ufl

from dolfinx import fem, log, mesh
from dolfinx.cpp.mesh import cell_num_entities
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import gmshio, VTXWriter
from mpi4py import MPI
from petsc4py import PETSc
from ufl import avg, div, dot, grad, inner, jump

import commons, mesh_utils, solvers, utils


LX = 75e-6
Wa_p = 1e3
kappa_elec = 0.1  # S/m
faraday_const = 96485
R = 8.3145
T = 298
i0_p = kappa_elec * R * T / (faraday_const * Wa_p * LX)
kr = 1
voltage = 1

Print = PETSc.Sys.Print
log.set_log_level(dolfinx.log.LogLevel.INFO)


def surface_overpotential(kappa, u, n, i0, kinetics_type='linear', ref={"L": 1, "phi": 1, "t": 1, "c": 1}):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='secondary current distribution')
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    args = parser.parse_args()

    markers = commons.Markers()
    comm = MPI.COMM_WORLD
    rank = comm.rank
    dtype = PETSc.ScalarType
    workdir = os.path.join(args.mesh_folder, "hdg")
    utils.make_dir_if_missing(workdir)
    output_meshfile = os.path.join(args.mesh_folder, 'mesh.msh')
    lines_h5file = os.path.join(args.mesh_folder, 'lines.h5')
    potential_resultsfile = os.path.join(workdir, "potential.bp")
    u_resultsfile = os.path.join(workdir, "u.bp")
    ubar_resultsfile = os.path.join(workdir, "ubar.bp")
    concentration_resultsfile = os.path.join(workdir, "concentration.bp")
    current_resultsfile = os.path.join(workdir, "current.bp")
    simulation_metafile = os.path.join(workdir, "simulation.json")

    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = gmshio.read_from_msh(output_meshfile, comm, partitioner=partitioner)[:3]
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)
    domain.topology.create_connectivity(tdim, tdim)
    domain.topology.create_connectivity(fdim, fdim)

    # tag internal facets as 0
    ft_imap = domain.topology.index_map(fdim)
    num_facets = ft_imap.size_local + ft_imap.num_ghosts
    indices = np.arange(0, num_facets)
    values = np.zeros(indices.shape, dtype=np.intc)
    values[ft.indices] = ft.values
    ft = mesh.meshtags(domain, fdim, indices, values)
    ct = mesh.meshtags(domain, tdim, ct.indices, ct.values)

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

    dS = ufl.Measure("dS", domain=domain, subdomain_data=int_facet_domains)

    # create submeshes
    # tag internal facets as 0
    ft_imap = domain.topology.index_map(fdim)
    num_facets_local = ft_imap.size_local + ft_imap.num_ghosts
    num_facets_local_se = num_facets_local
    num_facets_local_am = num_facets_local
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
    submesh_electrolyte.topology.create_entities(fdim)
    submesh_positive_am, submesh_positive_am_to_mesh, t_v_map = mesh.create_submesh(
        domain, tdim, ct.find(markers.positive_am)
    )[0:3]
    submesh_positive_am.topology.create_entities(fdim)
    parent_to_sub_electrolyte = np.full(num_facets_local_se, -1, dtype=np.int32)
    parent_to_sub_electrolyte[submesh_electrolyte_to_mesh] = np.arange(len(submesh_electrolyte_to_mesh), dtype=np.int32)
    parent_to_sub_positive_am = np.full(num_facets_local_am, -1, dtype=np.int32)
    parent_to_sub_positive_am[submesh_positive_am_to_mesh] = np.arange(len(submesh_positive_am_to_mesh), dtype=np.int32)

    ft_electrolyte = mesh_utils.transfer_meshtags(domain, submesh_electrolyte, submesh_electrolyte_to_mesh, ft)
    ft_positive_am = mesh_utils.transfer_meshtags(domain, submesh_positive_am, submesh_positive_am_to_mesh, ft)

    # Hack, as we use one-sided restrictions, pad dS integral with the same entity from the same cell on both sides
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(fdim, fdim)
    domain.topology.create_connectivity(tdim, fdim)
    f_to_c = domain.topology.connectivity(fdim, tdim)

    for facet in ft.find(markers.electrolyte_v_positive_am):
        cells = f_to_c.links(facet)
        assert len(cells) == 2
        b_map = parent_to_sub_electrolyte[cells]
        t_map = parent_to_sub_positive_am[cells]
        parent_to_sub_electrolyte[cells] = max(b_map)
        parent_to_sub_positive_am[cells] = max(t_map)

    entity_maps = {submesh_electrolyte: parent_to_sub_electrolyte, submesh_positive_am: parent_to_sub_positive_am}

    # facets submesh
    submesh_facets, submesh_facets_to_mesh = mesh.create_submesh(
    domain, fdim, ft.indices)[:2]
    parent_to_facets = np.full(num_facets_local, -1, dtype=np.int32)
    parent_to_facets[submesh_facets_to_mesh] = np.arange(len(submesh_facets_to_mesh), dtype=np.int32)
    entity_maps[submesh_facets] = parent_to_facets

    submesh_facets_se, submesh_facets_se_to_mesh = mesh.create_submesh(
        domain, fdim, ft_electrolyte.indices)[:2]
    parent_to_facets_se = np.full(num_facets_local_se, -1, dtype=np.int32)
    parent_to_facets_se[submesh_facets_se_to_mesh] = np.arange(len(submesh_facets_se_to_mesh), dtype=np.int32)
    entity_maps[submesh_facets_se] = parent_to_facets_se

    submesh_facets_am, submesh_facets_am_to_mesh = mesh.create_submesh(
        domain, fdim, ft_positive_am.indices)[:2]
    parent_to_facets_am = np.full(num_facets_local_am, -1, dtype=np.int32)
    parent_to_facets_am[submesh_facets_am_to_mesh] = np.arange(len(submesh_facets_am_to_mesh), dtype=np.int32)
    entity_maps[submesh_facets_am] = parent_to_facets_am

    submesh_electrolyte.topology.create_connectivity(fdim, tdim)
    submesh_electrolyte.topology.create_connectivity(tdim, fdim)
    submesh_positive_am.topology.create_connectivity(fdim, tdim)
    submesh_positive_am.topology.create_connectivity(tdim, fdim)

    # function spaces
    k = 2
    V = fem.functionspace(domain, ("DG", k))
    Vbar = fem.functionspace(submesh_facets, ("DG", k))
    V0 = fem.functionspace(submesh_electrolyte, ("DG", k))
    V0bar = fem.functionspace(submesh_facets_se, ("DG", k))
    V1 = fem.functionspace(submesh_positive_am, ("DG", k))
    V1bar = fem.functionspace(submesh_facets_am, ("DG", k))

    # Cell space
    u, v = fem.Function(V), ufl.TestFunction(V)
    ue, ve = fem.Function(V0), ufl.TestFunction(V0)
    up, vp = fem.Function(V1), ufl.TestFunction(V1)

    # Facet space
    ubar, vbar = fem.Function(Vbar), ufl.TestFunction(Vbar)
    uebar, vebar = fem.Function(V0bar), ufl.TestFunction(V0bar)
    upbar, vpbar = fem.Function(V1bar), ufl.TestFunction(V1bar)

    # Define integration measures
    dx = ufl.Measure("dx", domain=domain, subdomain_data=ct)
    dxe = ufl.Measure("dx", domain=submesh_electrolyte)
    dxp = ufl.Measure("dx", domain=submesh_positive_am)

    all_facets = mesh_utils.compute_cell_boundary_facets(domain, ct, [markers.electrolyte, markers.positive_am])
    se_facets = mesh_utils.compute_cell_boundary_facets(domain, ct, [markers.electrolyte])
    am_facets = mesh_utils.compute_cell_boundary_facets(domain, ct, [markers.positive_am])
    left_facets = mesh_utils.compute_interface_cell_boundary_facets(domain, ct, ft, markers.electrolyte, markers.left)
    right_facets = mesh_utils.compute_interface_cell_boundary_facets(domain, ct, ft, markers.positive_am, markers.right)
    se_x_facets = mesh_utils.compute_interface_cell_boundary_facets(domain, ct, ft, markers.electrolyte, markers.electrolyte_v_positive_am)
    am_x_facets = mesh_utils.compute_interface_cell_boundary_facets(domain, ct, ft, markers.positive_am, markers.electrolyte_v_positive_am)
    se_minus_x_facets = utils.delete_numpy_rows(se_facets, se_x_facets)
    am_minus_x_facets = utils.delete_numpy_rows(am_facets, am_x_facets)
    right_bndry_facets = np.array(right_facets).flatten()
    left_bndry_facets = np.array(left_facets).flatten()

    # Cell
    dx_c = ufl.Measure("dx", domain=domain, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    dse = ufl.Measure("ds", domain=submesh_electrolyte, subdomain_data=[(1, se_minus_x_facets.flatten()), (2, np.array(se_x_facets).flatten())])
    dsp = ufl.Measure("ds", domain=submesh_positive_am, subdomain_data=[(1, am_minus_x_facets.flatten()), (2, np.array(am_x_facets).flatten())])

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

    dSx = ufl.Measure("dS", domain=domain, subdomain_data=int_facet_domains, subdomain_id=markers.electrolyte_v_positive_am)

    n = ufl.FacetNormal(domain)
    ne = ufl.FacetNormal(submesh_electrolyte)
    np = ufl.FacetNormal(submesh_positive_am)
    h = ufl.CellDiameter(domain)
    he = ufl.CellDiameter(submesh_electrolyte)
    hp = ufl.CellDiameter(submesh_positive_am)
    gamma = 16.0 * k **2 / h  # Scaled penalty parameter
    gamma_e = 16.0 * k **2 / he  # Scaled penalty parameter
    gamma_p = 16.0 * k **2 / hp  # Scaled penalty parameter

    x = ufl.SpatialCoordinate(domain)

    left_boundary = ft_electrolyte.find(markers.left)
    right_boundary = ft_positive_am.find(markers.right)

    # Since the boundary condition is enforced in the facet space, we must
    # use the mesh_to_facet_mesh map to get the corresponding facets in
    # facet_mesh
    left_facet_mesh_boundary_facets = parent_to_facets_se[left_boundary]
    right_facet_mesh_boundary_facets = parent_to_facets_am[right_boundary]
    # Get the dofs and apply the bondary condition
    submesh_facets_se.topology.create_connectivity(fdim, fdim)
    submesh_facets_am.topology.create_connectivity(fdim, fdim)
    left_dofs = fem.locate_dofs_topological(V0bar, fdim, left_facet_mesh_boundary_facets)
    right_dofs = fem.locate_dofs_topological(V0bar, fdim, right_facet_mesh_boundary_facets)
    left_bc = fem.dirichletbc(dtype(0.0), left_dofs, V1bar)
    right_bc = fem.dirichletbc(dtype(voltage), right_dofs, V1bar)
    bcs = [left_bc, right_bc]

    # i_n = (-kappa * inner(grad(u), n))("+")
    # jump_u = R * T / (i0_p * faraday_const) * i_n
    # i_lin = i0_p * faraday_const / (R * T) * (u - ubar)
    alpha = 10
    # gamma = 10
    h_avg = avg(h)
    u_ocv = 0
    kr = 1
    kappa = 0.1
    kappa_e = kr * kappa
    kappa_p = kappa
    i0_p = 0.1 # [A/m2]

    # prescribed step of potential at charge transfer interface
    Print("Expression for prescribed step of potential..")
    u_step = surface_overpotential([kappa_e, kappa_p], [ue, up], [ne, np], i0_p)  + u_ocv #+ ocv_chen2020(c(r_res), cmax=c_max/c_ref)/phi_ref

    Print("Composing variational formulation")
    F0a = kappa_e * inner(grad(ue), grad(ve)) * dxe
    F0a += - kappa_e * inner(ue - uebar, inner(grad(ve), ne)) * dse(1)
    F0a += + kappa_e * inner(grad(ue), ne) * ve * dse(1)
    F0a += + gamma_e * kappa_e * inner(ue - uebar, ve) * dse(1)

    F0b = kappa_e * inner(grad(ue), ne) * vebar * dse(1)
    F0b += - gamma_e * kappa_e * inner(ue - uebar, vebar) * dse(1)

    F1a = kappa_p * inner(grad(up), grad(vp)) * dxp
    F1a += - kappa_p * inner(up - upbar, inner(grad(vp), np)) * dsp(1)
    F1a += + kappa_p * inner(grad(up), np) * vp * dsp(1)
    F1a += + gamma_p * kappa_p * inner(up - upbar, vp) * dsp(1)

    F1b = kappa_p * inner(grad(up), np) * vpbar * dsp(1)
    F1b += - gamma_p * kappa_p * inner(up - upbar, vpbar) * dsp(1)

    # add charge transfer coupling terms
    F0a += - 0.5 * inner(kappa_e * grad(ve), ne) * (up - ue - u_step) * dse(2)
    F0b += - gamma_e / 0.5 / (he + hp) * (up - ue - u_step) * vebar * dse(2)

    F1a += - 0.5 * inner(kappa_p * grad(vp), ne) * (up - ue - u_step) * dsp(2)
    F1b += + gamma_p / 0.5 / (he + hp) * (up - ue - u_step) * vpbar * dsp(2)

    jac00 = ufl.derivative(F0a, ue)
    jac01 = ufl.derivative(F0a, uebar)
    jac02 = ufl.derivative(F0a, up)
    jac03 = ufl.derivative(F0a, upbar)

    jac10 = ufl.derivative(F0b, ue)
    jac11 = ufl.derivative(F0b, uebar)
    jac12 = ufl.derivative(F0b, up)
    jac13 = ufl.derivative(F0b, upbar)

    jac20 = ufl.derivative(F1a, ue)
    jac21 = ufl.derivative(F1a, uebar)
    jac22 = ufl.derivative(F1a, up)
    jac23 = ufl.derivative(F1a, upbar)

    jac30 = ufl.derivative(F1b, ue)
    jac31 = ufl.derivative(F1b, uebar)
    jac32 = ufl.derivative(F1b, up)
    jac33 = ufl.derivative(F1b, upbar)

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

    J = [
            [J00, J01, J02, J03],
            [J10, J11, J12, J13],
            [J20, J21, J22, J23],
            [J30, J31, J32, J33],
        ]


    F = [
            fem.form(F0a, entity_maps=entity_maps),
            fem.form(F0b, entity_maps=entity_maps),
            fem.form(F1a, entity_maps=entity_maps),
            fem.form(F1b, entity_maps=entity_maps),
        ]

    Print("Composing solver..")

    # solver = scifem.NewtonSolver(
    #         F=F,
    #         J=J,
    #         w=[ue, uebar, up, upbar],
    #         bcs=bcs,
    #         # max_iterations=5,
    #         petsc_options={
    #         "ksp_type": "preonly",
    #         "pc_type": "lu",
    #         "pc_factor_mat_solver_type": "superlu_dist",
    #         },
    #         )
    Jmat = fem.petsc.create_matrix_block(J)
    Fvec = fem.petsc.create_vector_block(F)
    snes = PETSc.SNES().create(comm)
    snes.setType('newtonls')
    snes.setTolerances(rtol=1e-4, max_it=200)
    snes.setMonitor(lambda _, it, residual: Print("it:", it, "res:", residual))
    snes.getKSP().setType(PETSc.KSP.Type.FGMRES)
    snes.getKSP().getPC().setType(PETSc.PC.Type.ILU)
    snes.getKSP().setOptionsPrefix("snes_")
    # snes.getKSP().setOperators(Jmat, Jmat)
    snes.getKSP().setTolerances(rtol=1e-4)
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
    problem_t0 = solvers.NonlinearPDE_SNESProblem(F, J, [ue, uebar, up, upbar], bcs, P=J)
    snes.setFunction(problem_t0.F_block, Fvec)
    snes.setJacobian(problem_t0.J_block, J=Jmat, P=Jmat)
    x = fem.petsc.create_vector_block(F)
    x.set(0.0)
    t0 = time.time()
    snes.solve(None, x)
    t1 = time.time()
    snes.destroy()
    Jmat2d.destroy()
    Fvec2d.destroy()
    x2d.destroy()
    petsc_options.clear()
    # Print("Solving..")
    # solver.solve(1e-6)
    Print("Completed solve!")

    u.interpolate(ue, cells1=submesh_electrolyte_to_mesh, cells0=np.arange(len(submesh_electrolyte_to_mesh)))
    u.interpolate(up, cells1=submesh_positive_am_to_mesh, cells0=np.arange(len(submesh_positive_am_to_mesh)))
    u.x.scatter_forward()


    # Write to file
    with VTXWriter(domain.comm, u_resultsfile, u, "bp5") as f:
        f.write(0.0)

    # with VTXWriter(domain.comm, ubar_resultsfile, ubar, "bp5") as f:
    #     f.write(0.0)

    # interpolated functions
    W_DG = fem.functionspace(domain, ('DG', k))
    u_dg = fem.Function(W_DG)
    u_dg.interpolate(u)
    # W_CG = fem.functionspace(domain, ('CG', k, (3,)))
    # current_cg = fem.Function(W_CG)
    # current_expr = fem.Expression(-grad(u_dg), W_CG.element.interpolation_points())
    # current_cg.interpolate(current_expr)

    I_left = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(-kappa * grad(u_dg), n) * ds(markers.left))), op=MPI.SUM)
    I_middle = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(-(kappa * grad(u_dg))('+'), n('+')) * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)
    I_right = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(-kappa * grad(u_dg), n) * ds(markers.right))), op=MPI.SUM)
    I_insulated = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(inner(-kappa * grad(u_dg), n)) * ds(markers.insulated))), op=MPI.SUM)
    print(f"I_left       : {np.abs(I_left):.4e} A")
    print(f"I_middle     : {np.abs(I_middle):.4e} A")
    print(f"I_right      : {np.abs(I_right):.4e} A")
    print(f"I_insulated  : {np.abs(I_insulated):.4e} A")

    with VTXWriter(domain.comm, potential_resultsfile, u_dg, "bp5") as f:
        f.write(0.0)

    with VTXWriter(domain.comm, current_resultsfile, current_cg, "bp5") as f:
        f.write(0.0)
