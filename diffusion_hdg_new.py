#!/usr/bin/env python
# coding: utf-8



import argparse
import os
import sys
import time

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
from ufl import div, dot, grad, inner

import commons, solvers, solver_params, utils

LX = 75e-6
Wa_p = 1e-3
kappa_elec = 0.1  # S/m
faraday_const = 96485
R = 8.3145
T = 298
i0_p = kappa_elec * R * T / (faraday_const * Wa_p * LX)
kr = 1
voltage = 1


def compute_cell_boundary_facets(domain, ct, marker):
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

    return np.vstack((np.repeat(cells_1[perm], n_f), np.tile(np.arange(n_f), n_c))).T#.flatten()


def compute_interface_cell_boundary_facets(domain, ct, ft, cell_marker, facet_marker):
    """
    Compute integration entities for integrals around the boundaries of cells at the
    location of prescribed flux expression

    domain: the mesh
    ct: cell tags
    ft: facet tags
    cell_marker: marker for subdomain
    facet_marker: marker for interface
    """
    f_to_c = domain.topology.connectivity(fdim, tdim)
    c_to_f = domain.topology.connectivity(tdim, fdim)
    ft_imap = domain.topology.index_map(fdim)
    num_facets = ft_imap.size_local + ft_imap.num_ghosts
    interface_facets = ft.find(facet_marker)

    int_facet_domain = []
    lcells = []
    for f in interface_facets:
        if f >= ft_imap.size_local:# or len(f_to_c.links(f)) != 2:
            continue
        c_0 = f_to_c.links(f)[0]
        # c_0, c_1 = f_to_c.links(f)[0], f_to_c.links(f)[1]
        # subdomain_0, subdomain_1 = ct.values[[c_0, c_1]]
        local_f_0 = np.where(c_to_f.links(c_0) == f)[0][0]
        # local_f_1 = np.where(c_to_f.links(c_1) == f)[0][0]
        int_facet_domain.append([c_0, local_f_0])
        # if subdomain_0 == cell_marker:
        #     # lcells.append(c_0)
        #     int_facet_domain.append([c_0, local_f_0])
        #     continue
        # elif subdomain_1 == cell_marker:
        #     # lcells.append(c_1)
        #     int_facet_domain.append([c_1, local_f_1])
        #     continue
    # lcells = sorted(list(set(lcells)))

    # for cid in lcells:
    #     int_facet_domain.extend([[cid, idx] for idx in range(3)])

    return int_facet_domain


def delete_numpy_rows(in_arr, to_delete):
    out_arr = in_arr
    for row in to_delete:
        idx = np.where(np.all(out_arr == row, axis=1))[0][0]
        out_arr = np.delete(out_arr, idx, axis=0)

    return out_arr

markers = commons.Markers()
mesh_folder = "output/tertiary_current/75-40-0/unrefined/0.5/"
comm = MPI.COMM_WORLD
rank = comm.rank
dtype = PETSc.ScalarType
workdir = os.path.join(mesh_folder, "hdg")
utils.make_dir_if_missing(workdir)
output_meshfile = os.path.join(mesh_folder, 'mesh.msh')
lines_h5file = os.path.join(mesh_folder, 'lines.h5')
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

phase1 = 1
phase1_facets = compute_cell_boundary_facets(domain, ct, phase1)
left_facets = compute_interface_cell_boundary_facets(domain, ct, ft, phase1, markers.left)
right_facets = compute_interface_cell_boundary_facets(domain, ct, ft, phase1, markers.right)

out_arr1 = delete_numpy_rows(phase1_facets, left_facets)
out_arr2 = delete_numpy_rows(out_arr1, right_facets)
phase_1_facets = np.array(phase1_facets).flatten()
# non_bc_facets = np.vstack((out_arr1, out_arr2)).flatten()


# # Create the sub-mesh
# internal_dofs = np.array(list(set(ft.indices).difference(set(ft.find(markers.left)))), dtype=np.int32)
# print(internal_dofs.shape, ft.indices.shape)
facet_mesh, facet_mesh_to_mesh, _, _ = mesh.create_submesh(domain, fdim, ft.indices)
mesh_to_facet_mesh = np.full(num_facets, -1)
mesh_to_facet_mesh[facet_mesh_to_mesh] = np.arange(len(facet_mesh_to_mesh))
entity_maps = {facet_mesh: mesh_to_facet_mesh}

D = 1e-14
dt = 1e-4
# function spaces
k = 3
VC = fem.functionspace(domain, ("Discontinuous Lagrange", k))
VCbar = fem.functionspace(facet_mesh, ("Discontinuous Lagrange", k))


VC_map = VC.dofmap.index_map
VC_dofmap = VC.dofmap

VCbar_map = VCbar.dofmap.index_map
VCbar_dofmap = VCbar.dofmap

n_dofs_t0 = VC_map.size_global*VC.dofmap.index_map_bs + VCbar_map.size_global*VCbar.dofmap.index_map_bs

# Cell space
c, q = fem.Function(VC), ufl.TestFunction(VC)
c0 = fem.Function(VC)
c0.interpolate(lambda x: 100.0 + x[0] - x[0])
# Facet space
cbar, qbar = fem.Function(VCbar), ufl.TestFunction(VCbar)

# Define integration measures
# Cell
dx_c = ufl.Measure("dx", domain=domain, subdomain_data=ct)
# Cell boundaries
# We need to define an integration measure to integrate around the
# boundary of each cell.

ds_c = ufl.Measure("ds", subdomain_data=[(1, out_arr2.flatten()), (2, np.array(left_facets).flatten()), (3, np.array(right_facets).flatten())], domain=domain)
dS = ufl.Measure("dS", domain=domain, subdomain_data=ft)
ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
# Create a cell integral measure over the facet mesh
dx_f = ufl.Measure("dx", domain=facet_mesh, subdomain_data=ft)

h = ufl.CellDiameter(domain)
n = ufl.FacetNormal(domain)
gamma = 16.0 * k**2 / h  # Scaled penalty parameter

x = ufl.SpatialCoordinate(domain)


left_boundary = ft.find(markers.left)
right_boundary = ft.find(markers.right)

# Get the dofs and apply the bondary condition
# left_facet_mesh_boundary_facets = mesh_to_facet_mesh[left_boundary]
right_facet_mesh_boundary_facets = mesh_to_facet_mesh[right_boundary]
facet_mesh.topology.create_connectivity(fdim, fdim)
# left_dofs = fem.locate_dofs_topological(VCbar, fdim, left_facet_mesh_boundary_facets)
# left_bc = fem.dirichletbc(dtype(0.0), left_dofs, VCbar)
right_dofs = fem.locate_dofs_topological(VCbar, fdim, right_facet_mesh_boundary_facets)
right_bc = fem.dirichletbc(dtype(voltage), right_dofs, VCbar)
bcs = []#[right_bc]#, right_bc]

# ubar_right = fem.Function(VCbar)
# ubar_right.interpolate(lambda x: 1.0 + x[0]-x[0])
# with ubar_right.vector.localForm() as u0_loc:
#     u0_loc.set(voltage)

gbar = fem.Constant(facet_mesh, dtype(-1.333e-3))

F0 = (c - c0)/dt * q * dx_c
F0 += D * inner(grad(c), grad(q)) * dx_c
F0 += - D * inner(c - cbar, inner(grad(q), n)) * (ds_c(1) + ds_c(2) + ds_c(3))
F0 += + D * inner(grad(c), n) * q * (ds_c(1) + ds_c(2)+ds_c(3))
F0 += + gamma * D * inner(c - cbar, q) * (ds_c(1) + ds_c(2) + ds_c(3))

F1 = D * inner(grad(c), n) * qbar * (ds_c(1) + ds_c(2) + ds_c(3))
F1 += gamma * D * inner(c, qbar) * (ds_c(1) + ds_c(2) + ds_c(3))
F1 += -gamma * D * inner(cbar, qbar) * (ds_c(1) + ds_c(2) + ds_c(3))
F1 += gbar * qbar * (ds_c(2))

jac00 = ufl.derivative(F0, c)
jac01 = ufl.derivative(F0, cbar)

jac10 = ufl.derivative(F1, c)
jac11 = ufl.derivative(F1, cbar)

J00 = fem.form(jac00, entity_maps=entity_maps)
J01 = fem.form(jac01, entity_maps=entity_maps)

J10 = fem.form(jac10, entity_maps=entity_maps)
J11 = fem.form(jac11, entity_maps=entity_maps)

J = [[J00, J01], [J10, J11]]

F = [
        fem.form(F0, entity_maps=entity_maps),
        fem.form(F1, entity_maps=entity_maps),
        ]

# solver = solvers.NewtonSolver(
#         F,
#         J,
#         [c, cbar],
#         bcs=bcs,
#         max_iterations=1000,
#         petsc_options={
#         "ksp_type": "preonly",
#         "pc_type": "lu",
#         "pc_factor_mat_solver_type": "superlu_dist",
#         },
#         )
# t0 = time.time()
# solver.solve(1e-5)
# t1 = time.time()
Jmat2d = fem.petsc.create_matrix_block(J)
Fvec2d = fem.petsc.create_vector_block(F)
snes = PETSc.SNES().create(comm)
snes.setType('newtonls')
snes.setTolerances(rtol=2.5e-5, max_it=100)
snes.getKSP().setType(PETSc.KSP.Type.CG)
snes.getKSP().getPC().setType(PETSc.PC.Type.HYPRE)
snes.getKSP().setOptionsPrefix("snes_")
snes.getKSP().setOperators(Jmat2d, Jmat2d)
snes.getKSP().setTolerances(rtol=1e-8)
snes.setErrorIfNotConverged(True)
snes.getKSP().setErrorIfNotConverged(True)
snes.getKSP().setConvergenceHistory()
opts = PETSc.Options()
for optk, optv in solver_params.AMG_TYPES["hypre"].items():
        opts[f"{snes.getKSP().getOptionsPrefix()}{optk}"] = optv
opts['snes_linesearch_type'] = 'bt'
opts['snes_linesearch_monitor'] = None
opts['snes_monitor'] = None
# opts[f"{snes.getKSP().getOptionsPrefix()}pc_factor_levels"] = 0
# opts[f"{snes.getKSP().getOptionsPrefix()}pc_factor_fill"] = 1.0
snes.getKSP().setFromOptions()
snes.setFromOptions()
snes.view()

problem_t0 = solvers.NonlinearPDE_SNESProblem(F, J, [c, cbar], bcs, P=J)
snes.setFunction(problem_t0.F_block, Fvec2d)
snes.setJacobian(problem_t0.J_block, J=Jmat2d, P=Jmat2d)
x2d = fem.petsc.create_vector_block(F)
x2d.set(0.0)
t0 = time.time()
snes.solve(None, x2d)
t1 = time.time()
snes.destroy()
Jmat2d.destroy()
Fvec2d.destroy()
x2d.destroy()
PETSc.Sys.Print(f"Finished computation of initial (t = 0) potential distribution!\nn_dofs: {n_dofs_t0:,}\nsolve time: {t1 - t0:.3f}s")
# Write to file
with VTXWriter(domain.comm, u_resultsfile, [c], "bp5") as f:
    f.write(0.0)

with VTXWriter(domain.comm, ubar_resultsfile, [cbar], "bp5") as f:
    f.write(0.0)




# interpolated functions
W_DG = fem.functionspace(domain, ('DG', 1))
c_dg = fem.Function(W_DG)
c_dg.interpolate(c)
# W_CG = fem.functionspace(domain, ('CG', 1, (3,)))
# current_cg = fem.Function(W_CG)
# current_expr = fem.Expression(-D*grad(c_dg), W_CG.element.interpolation_points())
# current_cg.interpolate(current_expr)
I_left = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(-D * grad(c_dg), n) * ds(markers.left))), op=MPI.SUM)
I_middle = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(-(D * grad(c_dg))('+'), n('+')) * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)
I_right = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(-D * grad(c_dg), n) * ds(markers.right))), op=MPI.SUM)
I_insulated = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(inner(-D * grad(c_dg), n)) * ds(markers.insulated))), op=MPI.SUM)
PETSc.Sys.Print(f"I_left       : {np.abs(I_left):.4e} A")
PETSc.Sys.Print(f"I_middle     : {np.abs(I_middle):.4e} A")
PETSc.Sys.Print(f"I_right      : {np.abs(I_right):.4e} A")
PETSc.Sys.Print(f"I_insulated  : {np.abs(I_insulated):.4e} A")
# with VTXWriter(domain.comm, potential_resultsfile, c_dg, "bp5") as f:
#     f.write(0.0)

# with VTXWriter(domain.comm, current_resultsfile, current_cg, "bp5") as f:
#     f.write(0.0)




bb_trees = bb_tree(domain, domain.topology.dim)
n_points = 10000
tol = 1e-8  # Avoid hitting the outside of the domain
x = np.linspace(tol, 75e-6 - tol, n_points)
y = np.ones(n_points) * 0.5 * 40e-6  # midline
points = np.zeros((3, n_points))
points[0] = x
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
u_values = c_dg.eval(points_on_proc, cells)
fig, ax = plt.subplots()
ax.plot((1/1e-6) * points_on_proc[:, 0], u_values, "k", linewidth=2)
# ax.grid(True)
ax.axvline(x=25, linestyle='--', color='red', linewidth=0.5)
# ax.set_xlim([0, 75])
# ax.set_ylim([0, voltage])
ax.set_ylabel(r'$\phi$ [V]', rotation=0, labelpad=30, fontsize='xx-large')
ax.set_xlabel(r'x [$\mu$m]')
ax.set_title('Potential Across Midline')
ax.set_box_aspect(1);
ax.minorticks_on();
plt.tight_layout()
plt.show()






