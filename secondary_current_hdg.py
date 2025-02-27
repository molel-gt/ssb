#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import ufl

from dolfinx import fem, mesh

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

    # facets submesh
    submesh_facets_se, submesh_facets_se_to_mesh = mesh.create_submesh(
        domain, fdim, ft_electrolyte.indices)[:2]
    parent_to_facets_se = np.full(num_facets_local, -1, dtype=np.int32)
    parent_to_facets_se[submesh_facets_se_to_mesh] = np.arange(len(submesh_facets_se_to_mesh), dtype=np.int32)
    entity_maps[submesh_facets_se] = parent_to_facets_se

    submesh_facets_am, submesh_facets_am_to_mesh = mesh.create_submesh(
        domain, fdim, ft_positive_am.indices)[:2]
    parent_to_facets_am = np.full(num_facets_local, -1, dtype=np.int32)
    parent_to_facets_am[submesh_facets_am_to_mesh] = np.arange(len(submesh_facets_am_to_mesh), dtype=np.int32)
    entity_maps[submesh_facets_am] = parent_to_facets_am

    # function spaces
    k = 2
    V0 = fem.functionspace(submesh_electrolyte, ("DG", k))
    V0bar = fem.functionspace(submesh_facets_se, ("DG", k))
    V1 = fem.functionspace(submesh_positive_am, ("DG", k))
    V1bar = fem.functionspace(submesh_facets_se, ("DG", k))

    # Cell space
    u0, v0 = fem.Function(V0), ufl.TestFunction(V0)
    u1, v1 = fem.Function(V1), ufl.TestFunction(V1)

    # Facet space
    u0bar, v0bar = fem.Function(V0bar), ufl.TestFunction(V0bar)
    u1bar, v1bar = fem.Function(V1bar), ufl.TestFunction(V1bar)

    # Define integration measures
    dx0 = ufl.Measure("dx", domain=domain, subdomain_data=ct, subdomain_id=markers.electrolyte)
    dx1 = ufl.Measure("dx", domain=domain, subdomain_data=ct, subdomain_id=markers.positive_am)

    # Cell
    dx_c = ufl.Measure("dx", domain=domain, subdomain_data=ct)

    h = ufl.CellDiameter(domain)
    n = ufl.FacetNormal(domain)
    gamma = 16.0 * k**2 / h  # Scaled penalty parameter

    x = ufl.SpatialCoordinate(domain)




    left_boundary = ft.find(markers.left)
    right_boundary = ft.find(markers.right)

    # Since the boundary condition is enforced in the facet space, we must
    # use the mesh_to_facet_mesh map to get the corresponding facets in
    # facet_mesh
    left_facet_mesh_boundary_facets = mesh_to_facet_mesh[left_boundary]
    right_facet_mesh_boundary_facets = mesh_to_facet_mesh[right_boundary]
    # Get the dofs and apply the bondary condition
    facet_mesh.topology.create_connectivity(fdim, fdim)
    left_dofs = fem.locate_dofs_topological(V0bar, fdim, left_facet_mesh_boundary_facets)
    right_dofs = fem.locate_dofs_topological(V1bar, fdim, right_facet_mesh_boundary_facets)
    left_bc = fem.dirichletbc(dtype(0.0), left_dofs, V0bar)
    right_bc = fem.dirichletbc(dtype(voltage), right_dofs, V1bar)
    bcs = [left_bc, right_bc]

    i_n = (-kappa * inner(grad(u), n))("+")
    jump_u = R * T / (i0_p * faraday_const) * i_n
    i_lin = i0_p * faraday_const / (R * T) * (u - ubar)
    alpha = 10
    # gamma = 10
    h_avg = avg(h)
    u_ocv = 0

    F0 = kappa * inner(grad(u), grad(v)) * dx_c
    F0 += - kappa * inner(u - ubar, inner(grad(v), n)) * ds_c(1)
    F0 += + kappa * inner(grad(u), n) * v * ds_c(1)
    F0 += + gamma * kappa * inner(u - ubar, v) * ds_c(1)

    F1 = kappa * inner(grad(u), n) * vbar * ds_c(1)
    F1 += - gamma * kappa * inner(u - ubar, vbar) * ds_c(1)

    # F_2a = (c - c0)/dt * q * dx_r + inner(grad(c), grad(q)) * dx_r
    # F_2a += - inner(c - cbar, inner(grad(q), n_c)) * ds_c(99)
    # F_2a += + inner(grad(c), n_c) * q * ds_fc(99)
    # F_2a += inner(grad(c), n_c) * q * ds_fc(99)
    # F_2a = + gamma_r * inner(c - cbar, q) * ds_fc(99)

    # F_2b += inner(grad(c), n_c) * qbar * ds_fc(99)
    # F_2b += - gamma_r * inner(c - cbar, qbar) * ds_fc(99)



    jac00 = ufl.derivative(F0, u)
    jac01 = ufl.derivative(F0, ubar)

    jac10 = ufl.derivative(F1, u)
    jac11 = ufl.derivative(F1, ubar)

    J00 = fem.form(jac00, entity_maps=entity_maps)
    J01 = fem.form(jac01, entity_maps=entity_maps)

    J10 = fem.form(jac10, entity_maps=entity_maps)
    J11 = fem.form(jac11, entity_maps=entity_maps)

    J = [[J00, J01], [J10, J11]]


    F = [
            fem.form(F0, entity_maps=entity_maps),
            fem.form(F1, entity_maps=entity_maps),
            ]




    solver = solvers.NewtonSolver(
            F,
            J,
            [u, ubar],
            bcs=bcs,
            max_iterations=5,
            petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "superlu_dist",
            },
            )
    solver.solve(1e-6)




    # Write to file
    with VTXWriter(domain.comm, u_resultsfile, u, "bp5") as f:
        f.write(0.0)

    with VTXWriter(domain.comm, ubar_resultsfile, ubar, "bp5") as f:
        f.write(0.0)

    # interpolated functions
    W_DG = fem.functionspace(domain, ('DG', k))
    u_dg = fem.Function(W_DG)
    u_dg.interpolate(u)
    W_CG = fem.functionspace(domain, ('CG', k, (3,)))
    current_cg = fem.Function(W_CG)
    current_expr = fem.Expression(-grad(u_dg), W_CG.element.interpolation_points())
    current_cg.interpolate(current_expr)

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
