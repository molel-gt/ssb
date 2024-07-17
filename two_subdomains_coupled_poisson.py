#!/usr/bin/env python
import argparse
import json
import os
import timeit

import dolfinx
import gmsh
import numpy as np
import ufl
import warnings

from basix.ufl import element, mixed_element
from dolfinx import cpp, default_scalar_type, fem, graph, io, mesh, nls, plot
from dolfinx.fem import petsc
from dolfinx.graph import partitioner_parmetis
from dolfinx.io import gmshio, VTXWriter
from dolfinx.nls import petsc as petsc_nls
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 dot, div, dx, ds, dS, grad, inner, grad, avg, jump)

import commons, solvers, utils

warnings.simplefilter('ignore')

dtype = PETSc.ScalarType
kappa_elec = 0.1
faraday_const = 96485
R = 8.3145
T = 298
D = 1e-15


if __name__ == '__main__':
    phase_1 = 1
    phase_2 = 2

    left = 1
    bottom_left = 2
    bottom_right = 3
    right = 4
    top_right = 5
    top_left = 6
    middle = 7

    comm = MPI.COMM_WORLD
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = gmshio.read_from_msh("mesh.msh", comm, partitioner=partitioner)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)
    f_to_c = domain.topology.connectivity(fdim, tdim)
    c_to_f = domain.topology.connectivity(tdim, fdim)

    ft_imap = domain.topology.index_map(fdim)
    num_facets = ft_imap.size_local + ft_imap.num_ghosts
    indices = np.arange(0, num_facets)
    values = np.zeros(indices.shape, dtype=np.intc)  # all facets are tagged with zero

    values[ft.indices] = ft.values
    ft = mesh.meshtags(domain, fdim, indices, values)

    # create submesh
    submesh, entity_map, vertex_map, geom_map = mesh.create_submesh(
        domain, tdim, ct.find(phase_2)
    )
    # transfer tags from parent to submesh
    tdim = domain.topology.dim
    fdim = tdim - 1
    f_map = domain.topology.index_map(fdim)
    all_facets = f_map.size_local + f_map.num_ghosts
    all_values = np.zeros(all_facets, dtype=np.int32)
    all_values[ft.indices] = ft.values

    # submesh for phase 2
    submesh.topology.create_entities(fdim)
    subf_map = submesh.topology.index_map(fdim)
    submesh.topology.create_connectivity(tdim, fdim)
    submesh.topology.create_connectivity(tdim, tdim)
    submesh.topology.create_connectivity(fdim, fdim)
    c_to_f_sub = submesh.topology.connectivity(tdim, fdim)
    num_sub_facets = subf_map.size_local + subf_map.num_ghosts
    sub_values = np.empty(num_sub_facets, dtype=np.int32)
    for i, entity in enumerate(entity_map):
        parent_facets = c_to_f.links(entity)
        child_facets = c_to_f_sub.links(i)
        for child, parent in zip(child_facets, parent_facets):
            sub_values[child] = all_values[parent]
    submesh_ft = mesh.meshtags(submesh, submesh.topology.dim - 1, np.arange(
        num_sub_facets, dtype=np.int32), sub_values)
    submesh.topology.create_connectivity(submesh.topology.dim - 1, submesh.topology.dim)

    # entity_maps = {submesh: entity_map, domain: ct.indices}
    mesh_to_submesh = np.full(len(ct.indices), -1)
    mesh_to_submesh[entity_map] = np.arange(len(entity_map))
    # entity_maps = {submesh: mesh_to_submesh, domain: ct.indices}

    ### interface facet
    submeshf, submeshf_to_mesh = mesh.create_submesh(domain, fdim, ft.find(middle))[0:2]
    msh_to_submeshf = np.full(num_facets, -1)
    msh_to_submeshf[submeshf_to_mesh] = np.arange(len(submeshf_to_mesh))
    entity_maps = {submeshf: msh_to_submeshf}

    

    # integration measures
    dx = ufl.Measure("dx", domain=domain, subdomain_data=ct)
    dx_r = ufl.Measure("dx", domain=submesh)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    ds_r = ufl.Measure("ds", domain=submesh, subdomain_data=submesh_ft)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=ft)
    facet_integration_entities = []
    domain.topology.create_connectivity(tdim, fdim)
    domain.topology.create_connectivity(fdim, tdim)
    c_to_f = domain.topology.connectivity(tdim, fdim)
    f_to_c = domain.topology.connectivity(fdim, tdim)
    # Loop through all interface facets
    for facet in ft.find(middle):
        # Check if this facet is owned
        if facet < ft_imap.size_local:
            # Get a cell connected to the facet
            cell = f_to_c.links(facet)[0]
            local_facet = c_to_f.links(cell).tolist().index(facet)
            facet_integration_entities.extend([cell, local_facet])
    # ds = ufl.Measure("ds", domain=submesh, subdomain_data=[(middle, facet_integration_entities)])

    V0 = fem.functionspace(domain, ("DG", 1))
    V1 = fem.functionspace(domain, ("CG", 1))
    # Create function space for the Lagrange multiplier
    V2 = fem.functionspace(submeshf, ("DG", 1))
    lamda, eta = fem.Function(V2), ufl.TestFunction(V2)
    # dlamda = ufl.TrialFunction(V2)

    u = fem.Function(V0)
    c = fem.Function(V1)
    u0 = fem.Function(V0)
    c0 = fem.Function(V1)
    c0.interpolate(lambda x: 16000 + x[0] - x[0])
    c.interpolate(lambda x: 100 + x[0] - x[0])
    q = ufl.TestFunction(V1)
    v = ufl.TestFunction(V0)
    n = ufl.FacetNormal(domain)
    nc = ufl.FacetNormal(domain)
    x = ufl.SpatialCoordinate(domain)

    h = ufl.CellDiameter(domain)
    h_avg = avg(h)

    # constants
    f = fem.Constant(domain, dtype(0))
    fc = fem.Constant(domain, dtype(0))
    g = fem.Constant(domain, dtype(0))
    gc = fem.Constant(domain, dtype(0))
    # dirichlet bc
    u_left = fem.Function(V0)
    with u_left.vector.localForm() as u0_loc:
        u0_loc.set(0)
    u_right = fem.Function(V0)
    with u_right.vector.localForm() as u1_loc:
        u1_loc.set(1)

    # kappa varying in each domain
    kappa = 1
    D = 1e-15

    # variational formulation

    alpha = 10
    gamma = 10
    # solve coupled problem
    dt = 1e-3
    TIME = 5 * dt
    t = 0

    F0 = kappa * inner(grad(u), grad(v)) * dx - f * v * dx - kappa * inner(grad(u), n) * v * ds

    # # Add DG/IP terms
    F0 += - avg(kappa) * inner(jump(u, n), avg(grad(v))) * dS
    F0 += - inner(avg(kappa * grad(u)), jump(v, n)) * dS
    F0 += alpha / h_avg * avg(kappa) * inner(jump(v, n), jump(u, n)) * dS

    # Nitsche Dirichlet BC terms on left and right boundaries
    F0 += - kappa * (u - u_left) * inner(n, grad(v)) * ds(left)
    F0 += - gamma / h * (u - u_left) * v * ds(left)
    F0 += - kappa * (u - u_right) * inner(n, grad(v)) * ds(right) 
    F0 += - gamma / h * (u - u_right) * v * ds(right)

    # Nitsche Neumann BC terms on insulated boundary
    F0 += - g * v * (ds(top_left) + ds(bottom_left)) + gamma * h * g * inner(grad(v), n) * (ds(top_left) + ds(bottom_left))
    F0 += - gamma * h * inner(inner(grad(u), n), inner(grad(v), n)) * (ds(top_left) + ds(bottom_left))
    F0 += - g * v * (ds(top_right) + ds(bottom_right)) + gamma * h * g * inner(grad(v), n) * (ds(top_right) + ds(bottom_right))
    F0 += - gamma * h * inner(inner(grad(u), n), inner(grad(v), n)) * (ds(top_right) + ds(bottom_right))

    F1 = 1/dt * inner(c - c0, q) * dx
    F1 += inner(D * grad(c), grad(q)) * dx
    F1 += - fc * q * dx
    F1 += - gc * q * (ds(top_right) + ds(bottom_right) + ds(right))
    F1 += - gc * q * (ds(top_left) + ds(bottom_left))
    # set flux boundary at the interface
    F1 += - inner(D * grad(c), n) * q * ds(left)

    F2  = + inner(eta, kappa/faraday_const * inner(grad(u), n)) * ds(left)
    F2 += - inner(eta, D * inner(grad(c), n)) * ds(left)
    F2 += + inner(lamda, kappa/faraday_const*inner(grad(eta), n)) * ds(left)
    F2 += - inner(lamda, D * inner(grad(eta), n)) * ds(left)

    while t < TIME:
        print(f"Time: {t:.3f}")
        t += dt
        jac00 = ufl.derivative(F0, u)
        jac01 = ufl.derivative(F0, c)
        jac02 = ufl.derivative(F0, lamda)

        jac10 = ufl.derivative(F1, u)
        jac11 = ufl.derivative(F1, c)
        jac12 = ufl.derivative(F1, lamda)

        jac20 = ufl.derivative(F2, u)
        jac21 = ufl.derivative(F2, c)
        jac22 = ufl.derivative(F2, lamda)
        
        J00 = fem.form(jac00, entity_maps=entity_maps)
        J01 = fem.form(jac01, entity_maps=entity_maps)
        J02 = fem.form(jac02, entity_maps=entity_maps)
        J10 = fem.form(jac10, entity_maps=entity_maps)
        J11 = fem.form(jac11, entity_maps=entity_maps)
        J12 = fem.form(jac12, entity_maps=entity_maps)
        J20 = fem.form(jac20, entity_maps=entity_maps)
        J21 = fem.form(jac21, entity_maps=entity_maps)
        J22 = fem.form(jac22, entity_maps=entity_maps)
        
        # J = [[J00, J01], [J10, J11],]
        J = [[J00, J01, J02], [J10, J11, J12], [J20, J21, J22]]
        F = [fem.form(F0, entity_maps=entity_maps),
            fem.form(F1, entity_maps=entity_maps),
            fem.form(F2, entity_maps=entity_maps),
            ]

        solver = solvers.NewtonSolver(
            F,
            J,
            [u, c, lamda],
            bcs=[],
            max_iterations=10,
            petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "superlu_dist",
            },
            )
        solver.solve(1e-6, beta=1)
        c0.x.array[:] = c.x.array
        if np.isnan(solver.dx.norm(0)):
            break
        I_left = comm.allreduce(fem.assemble_scalar(fem.form(inner(-(kappa * grad(u)), n) * ds(left))), op=MPI.SUM)
        I_right = comm.allreduce(fem.assemble_scalar(fem.form(inner(-(kappa * grad(u)), n) * ds(right))), op=MPI.SUM)
        print(f"Left and right: {I_left:.2e}, {I_right:.2e}")
        I_middle_1 = comm.allreduce(fem.assemble_scalar(fem.form(inner(-(kappa * grad(u))("+"), n("+")) * dS(middle))), op=MPI.SUM)
        I_middle_2 = comm.allreduce(fem.assemble_scalar(fem.form(inner(-faraday_const * D * grad(c), nc) * ds_r(middle), entity_maps=entity_maps)), op=MPI.SUM)
        print(f"Middle {I_middle_1:.2e}, {I_middle_2:.2e}")
