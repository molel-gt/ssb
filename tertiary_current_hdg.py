#!/usr/bin/env python3
# - Solve Poisson's equation using an HDG scheme.

# +
import argparse
import os
import sys

import dolfinx
import numpy as np
import ufl

from dolfinx import fem, mesh
from dolfinx.cpp.mesh import cell_num_entities

from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.io import gmshio, VTXWriter
from mpi4py import MPI
from petsc4py import PETSc
from ufl import div, dot, grad, inner

import commons, utils


def compute_cell_boundary_facets_new(domain, ct, marker):
    """Compute the integration entities for integrals around the
    boundaries of all cells in domain.

    Parameters:
        domain: The mesh.
        ct: cell tags
        marker: physical group label

    Returns:
        Facets to integrate over, identified by ``(cell, local facet
        index)`` pairs.
    """
    tdim = domain.topology.dim
    fdim = tdim - 1
    n_f = cell_num_entities(domain.topology.cell_type, fdim)
    cells_1 = ct.find(marker)
    perm = np.argsort(cells_1)
    n_c = cells_1.shape[0]
    print(n_f, n_c)

    return np.vstack((np.repeat(cells_1[perm], n_f), np.tile(np.arange(n_f), n_c))).T.flatten()


def compute_cell_boundary_facets(domain):
    """Compute the integration entities for integrals around the
    boundaries of all cells in domain.

    Parameters:
        domain: The mesh.

    Returns:
        Facets to integrate over, identified by ``(cell, local facet
        index)`` pairs.
    """
    tdim = domain.topology.dim
    fdim = tdim - 1
    n_f = cell_num_entities(domain.topology.cell_type, fdim)
    
    n_c = domain.topology.index_map(tdim).size_local
    print(n_f, n_c)

    return np.vstack((np.repeat(np.arange(n_c), n_f), np.tile(np.arange(n_f), n_c))).T.flatten()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Current Collector.')
    # parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="lithium_metal_3d_cc_2d")
    # parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1.0, type=float)
    # parser.add_argument("--Wa_n", help="Wagna number for negative electrode: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e-3, type=float)
    # parser.add_argument("--Wa_p", help="Wagna number for positive electrode: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e3, type=float)
    # parser.add_argument("--kr", help="ratio of ionic to electronic conductivity", nargs='?', const=1, default=1, type=float)
    # parser.add_argument("--gamma", help="interior penalty parameter", nargs='?', const=1, default=15, type=float)
    # parser.add_argument("--atol", help="solver absolute tolerance", nargs='?', const=1, default=1e-12, type=float)
    # parser.add_argument("--rtol", help="solver relative tolerance", nargs='?', const=1, default=1e-9, type=float)
    # parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
    #                     const=1, default='MICRON_TO_METER', type=str)

    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.rank
    dtype = PETSc.ScalarType

    workdir = os.path.join(args.mesh_folder, "results")
    utils.make_dir_if_missing(workdir)
    output_meshfile = os.path.join(args.mesh_folder, 'mesh.msh')
    lines_h5file = os.path.join(args.mesh_folder, 'lines.h5')
    potential_resultsfile = os.path.join(workdir, "potential.bp")
    u_resultsfile = os.path.join(workdir, "u.bp")
    ubar_resultsfile = os.path.join(workdir, "ubar.bp")
    concentration_resultsfile = os.path.join(workdir, "concentration.bp")
    current_resultsfile = os.path.join(workdir, "current.bp")
    simulation_metafile = os.path.join(workdir, "simulation.json")

    markers = commons.Markers()

    # ### Read input geometry
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = gmshio.read_from_msh(output_meshfile, comm, partitioner=partitioner)
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

    # Create the sub-mesh
    facet_mesh, facet_mesh_to_mesh, _, _ = mesh.create_submesh(domain, fdim, ft.indices)

    # Define function spaces
    k = 3  # Polynomial order
    V = fem.functionspace(domain, ("Discontinuous Lagrange", k))
    Vbar = fem.functionspace(facet_mesh, ("Discontinuous Lagrange", k))

    # Trial and test functions
    # Cell space
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    # Facet space
    ubar, vbar = ufl.TrialFunction(Vbar), ufl.TestFunction(Vbar)

    # Define integration measures
    # Cell
    dx_c = ufl.Measure("dx", domain=domain, subdomain_data=ct)
    # Cell boundaries
    # We need to define an integration measure to integrate around the
    # boundary of each cell. The integration entities can be computed
    # using the following convenience function.
    cell_boundary_facets = compute_cell_boundary_facets(domain)
    cell_boundaries = 1  # A tag
    # Create the measure
    phase1 = markers.electrolyte
    phase2 = markers.positive_am
    phase1_facets = compute_cell_boundary_facets_new(domain, ct, phase1)
    phase2_facets = compute_cell_boundary_facets_new(domain, ct, phase2)

    ds_c = ufl.Measure("ds", subdomain_data=[(phase1, phase1_facets), (phase2, phase2_facets)], domain=domain)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=ft)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    # ds_c = ufl.Measure("ds", subdomain_data=[(cell_boundaries, cell_boundary_facets)], domain=domain)
    # Create a cell integral measure over the facet mesh
    dx_f = ufl.Measure("dx", domain=facet_mesh, subdomain_data=ft)

    # We write the mixed domain forms as integrals over domain. Hence, we must
    # provide a map from facets in domain to cells in facet_mesh. This is the
    # 'inverse' of facet_mesh_to_mesh, which we compute as follows:
    mesh_to_facet_mesh = np.full(num_facets, -1)
    mesh_to_facet_mesh[facet_mesh_to_mesh] = np.arange(len(facet_mesh_to_mesh))
    entity_maps = {facet_mesh: mesh_to_facet_mesh}

    # Define forms
    h = ufl.CellDiameter(domain)
    n = ufl.FacetNormal(domain)
    gamma = 16.0 * k**2 / h  # Scaled penalty parameter

    x = ufl.SpatialCoordinate(domain)
    c = 1.0
    a_00 = fem.form(
        inner(c * grad(u), grad(v)) * dx_c
        - (
            inner(c * u, dot(grad(v), n)) * (ds_c(phase1) + ds_c(phase2))
            + inner(dot(grad(u), n), c * v) * (ds_c(phase1) + ds_c(phase2))
        )
        + gamma * inner(c * u, v) * (ds_c(phase1) + ds_c(phase2))
    )
    a_10 = fem.form(
        inner(dot(grad(u), n) - gamma * u, c * vbar) * (ds_c(phase1) + ds_c(phase2)), entity_maps=entity_maps
    )
    a_01 = fem.form(
        inner(c * ubar, dot(grad(v), n) - gamma * v) * (ds_c(phase1) + ds_c(phase2)), entity_maps=entity_maps
    )
    a_11 = fem.form(gamma * inner(c * ubar, vbar) * (ds_c(phase1) + ds_c(phase2)), entity_maps=entity_maps)

    # Manufacture a source term
    f = fem.Constant(domain, dtype(0.0))

    L_0 = fem.form(inner(f, v) * dx_c)
    L_1 = fem.form(inner(fem.Constant(facet_mesh, dtype(0.0)), vbar) * dx_f)

    # Define block structure
    a = [[a_00, a_01], [a_10, a_11]]
    L = [L_0, L_1]

    # Apply Dirichlet boundary conditions
    # We begin by locating the boundary facets of domain
    left_boundary = ft.find(markers.left)
    right_boundary = ft.find(markers.right)

    # Since the boundary condition is enforced in the facet space, we must
    # use the mesh_to_facet_mesh map to get the corresponding facets in
    # facet_mesh
    left_facet_mesh_boundary_facets = mesh_to_facet_mesh[left_boundary]
    right_facet_mesh_boundary_facets = mesh_to_facet_mesh[right_boundary]
    # Get the dofs and apply the bondary condition
    facet_mesh.topology.create_connectivity(fdim, fdim)
    left_dofs = fem.locate_dofs_topological(Vbar, fdim, left_facet_mesh_boundary_facets)
    right_dofs = fem.locate_dofs_topological(Vbar, fdim, right_facet_mesh_boundary_facets)
    left_bc = fem.dirichletbc(dtype(0.0), left_dofs, Vbar)
    right_bc = fem.dirichletbc(dtype(args.voltage), right_dofs, Vbar)

    # Assemble the matrix and vector
    A = assemble_matrix_block(a, bcs=[left_bc, right_bc])
    A.assemble()
    b = assemble_vector_block(L, a, bcs=[left_bc, right_bc])

    # Setup the solver
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("superlu_dist")

    # Compute solution
    x = A.createVecRight()
    ksp.solve(b, x)

    # Create functions for the solution and update values
    u, ubar = fem.Function(V), fem.Function(Vbar)
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    u.x.array[:offset] = x.array_r[:offset]
    ubar.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]
    u.x.scatter_forward()
    ubar.x.scatter_forward()

    # Write to file
    with VTXWriter(domain.comm, u_resultsfile, u, "bp5") as f:
        f.write(0.0)

    with VTXWriter(domain.comm, ubar_resultsfile, ubar, "bp5") as f:
        f.write(0.0)

    # interpolated functions
    W_DG = fem.functionspace(domain, ('DG', 1))
    u_dg = fem.Function(W_DG)
    u_dg.interpolate(u)
    W_CG = fem.functionspace(domain, ('CG', 1, (3,)))
    current_cg = fem.Function(W_CG)
    current_expr = fem.Expression(-grad(u_dg), W_CG.element.interpolation_points())
    current_cg.interpolate(current_expr)
    I_left = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(grad(u_dg), n) * ds(markers.left))), op=MPI.SUM)
    I_middle = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(grad(u_dg)('+'), n('+')) * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)
    I_right = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(grad(u_dg), n) * ds(markers.right))), op=MPI.SUM)
    I_insulated = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(inner(grad(u_dg), n)) * ds(markers.insulated))), op=MPI.SUM)
    print(f"I_left       : {np.abs(I_left):.4e} A")
    print(f"I_middle     : {np.abs(I_middle):.4e} A")
    print(f"I_right      : {np.abs(I_right):.4e} A")
    print(f"I_insulated  : {np.abs(I_insulated):.4e} A")
    with VTXWriter(domain.comm, potential_resultsfile, u_dg, "bp5") as f:
        f.write(0.0)

    with VTXWriter(domain.comm, current_resultsfile, current_cg, "bp5") as f:
        f.write(0.0)
