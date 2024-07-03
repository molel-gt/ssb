#!/usr/bin/env python
import argparse
import json
import os
import timeit

import dolfinx
import numpy as np
import petsc4py
import ufl
import warnings

from dolfinx import cpp, default_scalar_type, fem, graph, io, mesh, nls, plot
from dolfinx.fem import petsc
from dolfinx.io import gmshio, VTXWriter
from dolfinx.nls import petsc as petsc_nls
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 dot, div, dx, ds, dS, grad, inner, grad, avg, jump)

from restriction import Restriction
warnings.simplefilter('ignore')

dtype = PETSc.ScalarType

faraday_const = 96485
D = 1e-15
kappa = 0.1


if __name__ == '__main__':
    phase_1 = 1
    phase_2 = 2

    left = 1
    right = 4
    middle = 7
    external = 8
    insulated_phase_1 = 9
    insulated_phase_2 = 10
    output_meshfile = 'mesh.msh'
    # output_meshfile = "output/tertiary_current/150-40-0/20-55-20/5.0e-06/mesh.msh"
    comm = MPI.COMM_WORLD
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = gmshio.read_from_msh(output_meshfile, comm, partitioner=partitioner)
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)

    ft_imap = domain.topology.index_map(fdim)
    num_facets = ft_imap.size_local + ft_imap.num_ghosts
    indices = np.arange(0, num_facets, dtype=np.int32)
    values = np.zeros(indices.shape, dtype=np.int32)  # all facets are tagged with zero

    values[ft.indices] = ft.values
    ft = mesh.meshtags(domain, fdim, indices, values)

    # create submesh
    submesh, sub_to_parent, vertex_map, geom_map = mesh.create_submesh(
        domain, tdim, ct.find(phase_2)
    )
    submesh.topology.create_connectivity(tdim, tdim)
    submesh.topology.create_connectivity(fdim, tdim)
    # transfer tags from parent to submesh
    tdim = domain.topology.dim
    fdim = tdim - 1

    c_imap = domain.topology.index_map(tdim)
    num_cells = c_imap.size_local + c_imap.num_ghosts
    mesh_to_submesh = np.full(num_cells, -1, dtype=np.int32)
    mesh_to_submesh[sub_to_parent] = np.arange(len(sub_to_parent), dtype=np.int32)
    entity_maps = {submesh: mesh_to_submesh}

    # interface mesh
    submeshf, subf_to_parent, _, _ = mesh.create_submesh(
        domain, fdim, ft.find(middle)
    )
    submeshf.topology.create_connectivity(fdim, fdim)
    mesh_to_facet_mesh = np.full(num_facets, -1, dtype=np.int32)
    mesh_to_facet_mesh[subf_to_parent] = np.arange(len(subf_to_parent), dtype=np.int32)
    entity_maps[submeshf] = mesh_to_facet_mesh

    
    # transfer tags from parent to submesh
    tdim = domain.topology.dim
    fdim = tdim - 1
    c_to_f = domain.topology.connectivity(tdim, fdim)
    f_map = domain.topology.index_map(fdim)
    all_facets = f_map.size_local + f_map.num_ghosts
    all_values = np.zeros(all_facets, dtype=np.int32)
    all_values[ft.indices] = ft.values
    # submesh facets
    c_to_f = domain.topology.connectivity(tdim, fdim)
    submesh.topology.create_entities(fdim)
    subf_map = submesh.topology.index_map(fdim)
    c_to_f_sub = submesh.topology.connectivity(tdim, fdim)
    num_sub_facets = subf_map.size_local + subf_map.num_ghosts
    sub_values = np.empty(num_sub_facets, dtype=np.int32)
    for i, entity in enumerate(sub_to_parent):
        parent_facets = c_to_f.links(entity)
        child_facets = c_to_f_sub.links(i)
        for child, parent in zip(child_facets, parent_facets):
            sub_values[child] = all_values[parent]
    submesh_ft = mesh.meshtags(submesh, submesh.topology.dim - 1, np.arange(
        num_sub_facets, dtype=np.int32), sub_values)
    submesh.topology.create_connectivity(submesh.topology.dim - 1, submesh.topology.dim)

    # others
    f_to_c = domain.topology.connectivity(fdim, tdim)
    c_to_f = domain.topology.connectivity(tdim, fdim)
    charge_xfer_facets = ft.find(middle)

    int_facet_domain = []
    interface_facet_domain = []
    for f in charge_xfer_facets:
        if f >= ft_imap.size_local or len(f_to_c.links(f)) != 2:
            continue
        c_0, c_1 = f_to_c.links(f)[0], f_to_c.links(f)[1]
        subdomain_0, subdomain_1 = ct.values[[c_0, c_1]]
        local_f_0 = np.where(c_to_f.links(c_0) == f)[0][0]
        local_f_1 = np.where(c_to_f.links(c_1) == f)[0][0]
        if subdomain_0 == phase_2:
            interface_facet_domain.extend([c_0, local_f_0])
        if subdomain_1 == phase_2:
            interface_facet_domain.extend([c_1, local_f_1])
        if subdomain_0 > phase_1:
            int_facet_domain.append(c_0)
            int_facet_domain.append(local_f_0)
            int_facet_domain.append(c_1)
            int_facet_domain.append(local_f_1)
        else:
            int_facet_domain.append(c_1)
            int_facet_domain.append(local_f_1)
            int_facet_domain.append(c_0)
            int_facet_domain.append(local_f_0)
    int_facet_domains = [(middle, int_facet_domain)]

    # integration measures
    dx = ufl.Measure("dx", domain=domain, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=ft)

    ds_x = ufl.Measure("ds", domain=domain, subdomain_data=submesh_ft)

    ds_r = ufl.Measure("ds", domain=submesh, subdomain_data=submesh_ft)
    dx_f = ufl.Measure('dx', domain=submeshf)
    # ds_f = ufl.Measure('ds', domain=submeshf)
    ds_f = ufl.Measure('ds', domain=domain, subdomain_data=[(middle, interface_facet_domain)])

    # Function Spaces
    k = 2
    V0 = fem.functionspace(domain, ("CG", k))  # whole domain
    V1 = fem.functionspace(submesh, ("CG", k))  # phase 2 subdomain
    V2 = fem.functionspace(submeshf, ("DG", k))  # middle boundary facets

    u = ufl.TrialFunction(V0)
    v = ufl.TestFunction(V0)
    c = ufl.TrialFunction(V1)
    q = ufl.TestFunction(V1)
    lamda = ufl.TrialFunction(V2)
    eta = ufl.TestFunction(V2)

    # initial condition
    c0 = fem.Function(V1)
    c0.interpolate(lambda x: 0.5 * 27000 + x[0] - x[0])

    n = ufl.FacetNormal(domain)
    nc = ufl.FacetNormal(submesh)
    x = ufl.SpatialCoordinate(domain)

    # constants
    dt_ = 1e-1
    dt = fem.Constant(submesh, dtype(dt_))
    f0 = fem.Constant(domain, dtype(0))
    f1 = fem.Constant(submesh, dtype(0))
    f2 = fem.Constant(submeshf, dtype(0))
    g0 = fem.Constant(domain, dtype(0))
    g1 = fem.Constant(submesh, dtype(0))
    u_left = fem.Function(V0)
    with u_left.vector.localForm() as u0_loc:
        u0_loc.set(0)
    u_right = fem.Function(V0)
    with u_right.vector.localForm() as u1_loc:
        u1_loc.set(3.5)

    # variational formulation
    a00 = fem.form(kappa * inner(grad(u), grad(v)) * dx)# + inner(-kappa*grad(u), n) * v * ds_f(middle))
    a01 = None
    a02 = fem.form(inner(lamda, kappa/faraday_const*inner(grad(v), n)) * ds_f(middle), entity_maps=entity_maps)

    a10 = fem.form(dt * inner(kappa/faraday_const * grad(u), n) * q * ds_f(middle), entity_maps=entity_maps)
    a11 = fem.form(inner(c, q) * dx(phase_2) - dt * inner(D * grad(c), grad(q)) * dx(phase_2) - dt * D * inner(grad(c), nc) * q * ds_f(middle), entity_maps=entity_maps)
    a12 = fem.form(- inner(lamda, D * inner(grad(q), nc)) * ds_f(middle), entity_maps=entity_maps)

    a20 = fem.form(inner(eta, kappa/faraday_const * inner(grad(u), n)) * ds_f(middle), entity_maps=entity_maps)
    a21 = fem.form(- inner(eta, D * inner(grad(c), nc)) * ds_f(middle), entity_maps=entity_maps)
    a22 = None

    left_bc = fem.dirichletbc(u_left, fem.locate_dofs_topological(V0, 1, ft.find(left)))
    right_bc = fem.dirichletbc(u_right, fem.locate_dofs_topological(V0, 1, ft.find(right)))

    a = [
            [a00, a01, a02],
            [a10, a11, a12],
            [a20, a21, a22]
        ]
    L0_ = inner(f0, v) * dx
    L0_ += inner(g0, v) * ds(insulated_phase_1)
    L0_ += inner(g0, v) * ds(insulated_phase_2)
    L0 = fem.form(L0_)

    L1_ = dt * inner(f1, q) * dx(phase_2) 
    L1_ += dt * inner(c0, q) * dx(phase_2)
    L1_ += dt * inner(g1, q) * ds(insulated_phase_2)
    L1_ += dt * inner(g1, q) * ds(right)
    L1 = fem.form(L1_, entity_maps=entity_maps)
    L2 = fem.form(inner(f2, eta) * ds_f(middle), entity_maps=entity_maps)
    L = [L0, L1, L2]

    # A.view()

    # solve coupled
    uh = fem.Function(V0)
    ch = fem.Function(V1)
    lamdah = fem.Function(V2)
    uvtx = VTXWriter(comm, "u.bp", [uh], "BP5")
    cvtx = VTXWriter(comm, "c.bp", [ch], "BP5")
    lvtx = VTXWriter(comm, "lamda.bp", [lamdah], "BP5")


    TIME = 10 * dt_
    t = 0
    vol = comm.allreduce(fem.assemble_scalar(fem.form(1 * dx(phase_2), entity_maps=entity_maps)), op=MPI.SUM)

    while t < TIME:
        print(f"Time: {t:.3f}")
        t += dt_

        A = fem.petsc.assemble_matrix_block(a, bcs=[left_bc, right_bc])
        A.assemble()
        b = fem.petsc.assemble_vector_block(L, a, bcs=[left_bc, right_bc])
        facets = np.hstack((ft.find(external), ft.find(middle)))
        dofs0 = fem.locate_dofs_topological(V0, tdim, ct.indices)
        dofs1 = fem.locate_dofs_topological(V1, tdim, ct.find(phase_2))
        dofs2 = fem.locate_dofs_topological(V1, fdim, ft.find(middle))
        restrict = Restriction(function_spaces=[V0, V1, V2], blocal_dofs=[dofs0, dofs1, dofs2])
        A = restrict.restrict_matrix(A)
        b = restrict.restrict_vector(b)

        ksp = PETSc.KSP().create(comm)
        ksp.setMonitor(lambda _, it, residual: print(it, residual))
        ksp.setOperators(A)
        opts = PETSc.Options()
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("superlu_dist")

        # Compute solution
        x = A.createVecLeft()
        ksp.solve(b, x)

        # Recover solution
        u, c, lmbda = fem.Function(V0), fem.Function(V1), fem.Function(V2)
        

        offset = V0.dofmap.index_map.size_local * V0.dofmap.index_map_bs
        offset2 = V1.dofmap.index_map.size_local * V1.dofmap.index_map_bs
        offset3 = V2.dofmap.index_map.size_local * V2.dofmap.index_map_bs
        print(offset, offset2, offset3)
        # print(x.array_r[:offset])

        uh.x.array[:offset] = x.array_r[:offset]
        uh.x.scatter_forward()

        ch.x.array[:offset2] = x.array_r[offset:offset+offset2]
        ch.x.scatter_forward()

        c0.x.array[:] = ch.x.array
        lamdah.x.array[:(len(x.array_r) - offset - offset2)] = x.array_r[offset + offset2:]
        lamdah.x.scatter_forward()

        uvtx.write(t)
        cvtx.write(t)
        lvtx.write(t)

        I_middle_1 = comm.allreduce(fem.assemble_scalar(fem.form(inner(-(kappa * grad(uh)), n) * ds_f(middle))), op=MPI.SUM)
        I_middle_2 = comm.allreduce(fem.assemble_scalar(fem.form(inner(-faraday_const * D * grad(ch), n) * ds_f(middle),entity_maps=entity_maps)), op=MPI.SUM)
        print(f"I_middle_1: {I_middle_1:.2e}, I_middle_2: {I_middle_2:.2e}")
    uvtx.close()
    cvtx.close()
    lvtx.close()
