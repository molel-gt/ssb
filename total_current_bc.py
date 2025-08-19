#!/usr/bin/env python3
import argparse
import time

import gmsh
import numpy as np
import pyvista as pv
import scifem
import ufl

from dolfinx import cpp, default_scalar_type, fem, mesh, plot
from dolfinx.io import gmshio, VTXWriter
from mpi4py import MPI
from petsc4py import PETSc
from ufl import inner, grad


class SolverTypes:
    def __init__(self):
        pass

    @property
    def direct(self):
        return "direct"

    @property
    def nested_iterative(self):
        return "nested_iterative"

    @property
    def block_iterative(self):
        return "block_iterative"

    @property
    def block_gs(self):
        return "block_gs"


class Boundaries:
    def __init__(self):
        pass

    @property
    def left(self):
        return 1

    @property
    def bottom(self):
        return 2

    @property
    def right(self):
        return 3

    @property
    def top(self):
        return 4

    @property
    def insulated(self):
        return 5

    @property
    def domain(self):
        return 5


def build_mesh(output_path, markers, Lx=100e-6, Ly=100e-6, resolution=1e-6):
    """
    generate mesh for given dimensions (`Lx` `Ly`) and write output to `output_path`
    """
    gmsh.initialize()
    gmsh.model.add('2D')
    gmsh.option.setNumber("Mesh.MeshSizeMax", resolution)
    coords = [
    (0, 0, 0),
    (Lx, 0, 0),
    (1.5*Lx, 0.5*Ly, 0),
    (Lx, Ly, 0),
    (0, Ly, 0),
    ]
    points = []
    lines = []
    # adding coordinates
    for coord in coords:
        points.append(gmsh.model.occ.addPoint(*coord))

    # adding lines
    for idx in range(-1, len(points)-1):
        lines.append(gmsh.model.occ.addLine(points[idx], points[idx+1]))

    gmsh.model.occ.synchronize()

    # connect lines into loop
    loop = gmsh.model.occ.addCurveLoop(lines)

    # create surface from loop
    surf = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()

    # add boundary markers
    gmsh.model.addPhysicalGroup(1, [lines[0]], markers.left, "left")
    gmsh.model.addPhysicalGroup(1, [lines[1]], markers.bottom, "bottom")
    gmsh.model.addPhysicalGroup(1, [lines[2], lines[3]], markers.right, "right")
    gmsh.model.addPhysicalGroup(1, [lines[4]], markers.top, "top")
    gmsh.model.addPhysicalGroup(1, [lines[1], lines[4]], markers.insulated, "insulated")
    gmsh.model.addPhysicalGroup(2, [surf], markers.domain, "domain")
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(output_path)
    gmsh.finalize()

    return 0


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
    n_f = cpp.mesh.cell_num_entities(domain.topology.cell_type, fdim)

    cells_1 = ct.find(marker)
    perm = np.argsort(cells_1)
    n_c = cells_1.shape[0]

    return np.vstack((np.repeat(cells_1[perm], n_f), np.tile(np.arange(n_f), n_c))).T


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
    tdim = domain.topology.dim
    fdim = tdim - 1
    f_to_c = domain.topology.connectivity(fdim, tdim)
    c_to_f = domain.topology.connectivity(tdim, fdim)
    ft_imap = domain.topology.index_map(fdim)
    num_facets = ft_imap.size_local + ft_imap.num_ghosts
    interface_facets = ft.find(facet_marker)

    int_facet_domain = []
    lcells = []
    for f in interface_facets:
        if f >= ft_imap.size_local:
            continue
        c_0 = f_to_c.links(f)[0]
        subdomain_0 = ct.values[c_0]
        local_f_0 = np.where(c_to_f.links(c_0) == f)[0][0]
        if subdomain_0 == cell_marker:
            int_facet_domain.append([c_0, local_f_0])

        if len(f_to_c.links(f)) == 2:
            c_1 = f_to_c.links(f)[1]
            subdomain_1 = ct.values[c_1]
            if subdomain_1 == cell_marker:
                local_f_1 = np.where(c_to_f.links(c_1) == f)[0][0]
                int_facet_domain.append([c_1, local_f_1])

    return int_facet_domain


def delete_numpy_rows(in_arr, to_delete):
    out_arr = in_arr
    for row in to_delete:
        idx = np.where(np.all(out_arr == row, axis=1))[0][0]
        out_arr = np.delete(out_arr, idx, axis=0)

    return out_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='secondary current distribution')
    parser.add_argument("-current", "--current", help="current boundary condition [A]", nargs='?', const=1, default=-1e-4, type=float)
    parser.add_argument('--solver_type', help='solver type to use', nargs='?',
                        const=1, default='direct', type=str)
    parser.add_argument("--regenerate_mesh", help="whether to regenerate mesh", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    solver_types = SolverTypes()
    markers = Boundaries()
    output_mesh_path = 'mesh.msh'
    output_potential_path = 'potential.bp'
    if args.regenerate_mesh:
        build_mesh(output_mesh_path, markers, Lx=100e-6, Ly=100e-6, resolution=1e-6)

    comm = MPI.COMM_WORLD
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = gmshio.read_from_msh(output_mesh_path, comm, partitioner=partitioner)[:3]
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)

    left_boundary = ft.find(markers.left)
    bottom_boundary = ft.find(markers.bottom)
    right_boundary = ft.find(markers.right)
    top_boundary = ft.find(markers.top)

    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)

    V = fem.functionspace(domain, ("Lagrange", 2))

    x = ufl.SpatialCoordinate(domain)
    n = ufl.FacetNormal(domain)

    # left facets submesh
    submesh_facets_left, submesh_facets_left_to_mesh = mesh.create_submesh(
        domain, fdim, ft.find(markers.left))[:2]
    num_facets_local = (
        domain.topology.index_map(fdim).size_local + domain.topology.index_map(fdim).num_ghosts
    )
    parent_to_facets_left = np.full(num_facets_local, -1, dtype=np.int32)
    parent_to_facets_left[submesh_facets_left_to_mesh] = np.arange(len(submesh_facets_left_to_mesh), dtype=np.int32)
    entity_maps = {submesh_facets_left: parent_to_facets_left}

    # right facets submesh
    submesh_facets_right, submesh_facets_right_to_mesh, f_v_map = mesh.create_submesh(
        domain, fdim, ft.find(markers.right))[:3]
    num_facets_local = (
        domain.topology.index_map(fdim).size_local + domain.topology.index_map(fdim).num_ghosts
    )
    parent_to_facets_right = np.full(num_facets_local, -1, dtype=np.int32)
    parent_to_facets_right[submesh_facets_right_to_mesh] = np.arange(len(submesh_facets_right_to_mesh), dtype=np.int32)
    entity_maps[submesh_facets_right] = parent_to_facets_right

    all_facets = compute_cell_boundary_facets(domain, ct, markers.domain)
    right_facets = compute_interface_cell_boundary_facets(domain, ct, ft, markers.domain, markers.right)
    left_facets = compute_interface_cell_boundary_facets(domain, ct, ft, markers.domain, markers.left)
    minus_right_facets = delete_numpy_rows(all_facets, right_facets)
    right_bndry_facets = np.array(right_facets).flatten()
    left_bndry_facets = np.array(left_facets).flatten()

    V_l = fem.functionspace(submesh_facets_left, ("Lagrange", 1))
    V_r = fem.functionspace(submesh_facets_right, ("Lagrange", 1))

    # # Create the measure
    dx = ufl.Measure('dx', domain=domain, subdomain_data=ct, subdomain_id=markers.domain)
    ds_c = ufl.Measure("ds", subdomain_data=[(1, minus_right_facets.flatten()), (2, left_bndry_facets), (3, right_bndry_facets)], domain=domain)

    R_right = scifem.create_real_functionspace(submesh_facets_right)

    u, du = fem.Function(V), ufl.TestFunction(V)
    lmbda1, mu1 = fem.Function(V_l), ufl.TestFunction(V_l)
    lmbda2, mu2 = fem.Function(V_r), ufl.TestFunction(V_r)

    V_cell, w = fem.Function(R_right), ufl.TestFunction(R_right)

    kappa = fem.Constant(domain, default_scalar_type(1.0))
    I_tot = fem.Constant(submesh_facets_right, PETSc.ScalarType(args.current))
    L_left = comm.allreduce(fem.assemble_scalar(fem.form(1 * ds(markers.left))), op=MPI.SUM)
    L_bottom = comm.allreduce(fem.assemble_scalar(fem.form(1 * ds(markers.bottom))), op=MPI.SUM)
    L_right = comm.allreduce(fem.assemble_scalar(fem.form(1 * ds(markers.right))), op=MPI.SUM)
    L_top = comm.allreduce(fem.assemble_scalar(fem.form(1 * ds(markers.top))), op=MPI.SUM)

    V_map = V.dofmap.index_map
    V_dofmap = V.dofmap
    R_map = R_right.dofmap.index_map
    R_dofmap = R_right.dofmap
    V_l_map = V_l.dofmap.index_map
    V_l_dofmap = V_l.dofmap
    V_r_map = V_r.dofmap.index_map
    V_r_dofmap = V_r.dofmap
    n_dofs = V_map.size_global*V.dofmap.index_map_bs + R_map.size_global*R_right.dofmap.index_map_bs \
        + V_l_map.size_global*V_l.dofmap.index_map_bs + V_r_map.size_global*V_r.dofmap.index_map_bs
    F0 = kappa * inner(grad(u), grad(du)) * dx - du * lmbda1 * ds_c(2)  - du * lmbda2 * ds_c(3)
    F1 = u * mu1 * ds_c(2)
    F2 = (V_cell - u) * mu2 * ds_c(3)
    F3 = w * (I_tot/L_right + lmbda2) * ds_c(3)

    F = [
        fem.form(F0, entity_maps=entity_maps),
        fem.form(F1, entity_maps=entity_maps),
        fem.form(F2, entity_maps=entity_maps),
        fem.form(F3, entity_maps=entity_maps)
        ]
    j00 = ufl.derivative(F0, u)
    j01 = ufl.derivative(F0, lmbda1)
    j02 = ufl.derivative(F0, lmbda2)
    j03 = ufl.derivative(F0, V_cell)
    
    j10 = ufl.derivative(F1, u)
    j11 = ufl.derivative(F1, lmbda1)
    j12 = ufl.derivative(F1, lmbda2)
    j13 = ufl.derivative(F1, V_cell)

    j20 = ufl.derivative(F2, u)
    j21 = ufl.derivative(F2, lmbda1)
    j22 = ufl.derivative(F2, lmbda2)
    j23 = ufl.derivative(F2, V_cell)

    j30 = ufl.derivative(F3, u)
    j31 = ufl.derivative(F3, lmbda1)
    j32 = ufl.derivative(F3, lmbda2)
    j33 = ufl.derivative(F3, V_cell)

    J00 = fem.form(j00, entity_maps=entity_maps)
    J01 = fem.form(j01, entity_maps=entity_maps)
    J02 = fem.form(j02, entity_maps=entity_maps)
    J03 = fem.form(j03, entity_maps=entity_maps)

    J10 = fem.form(j10, entity_maps=entity_maps)
    J11 = fem.form(j11, entity_maps=entity_maps)
    J12 = fem.form(j12, entity_maps=entity_maps)
    J13 = fem.form(j13, entity_maps=entity_maps)

    J20 = fem.form(j20, entity_maps=entity_maps)
    J21 = fem.form(j21, entity_maps=entity_maps)
    J22 = fem.form(j22, entity_maps=entity_maps)
    J23 = fem.form(j23, entity_maps=entity_maps)

    J30 = fem.form(j30, entity_maps=entity_maps)
    J31 = fem.form(j31, entity_maps=entity_maps)
    J32 = fem.form(j32, entity_maps=entity_maps)
    J33 = fem.form(j33, entity_maps=entity_maps)

    J = [
        [J00, J01, J02, J03],
        [J10, J11, J12, J13],
        [J20, J21, J22, J23],
        [J30, J31, J32, J33],
        ]

    if args.solver_type == solver_types.direct:
        opts = {
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                'pc_factor_mat_solver_type': 'mumps',
                }
        solver = scifem.NewtonSolver(F, J, [u, lmbda1, lmbda2, V_cell], bcs=[], petsc_options=opts)
        PETSc.Sys.Print(f"Solving problem with total current condition of {np.abs(I_tot.value):.3f} [A], n_dofs: {n_dofs:,}")
        t0 = time.time()
        solver.solve()
        t1 = time.time()
        PETSc.Sys.Print(f"Solve time: {t1 - t0:.3f}s, n_dofs: {n_dofs:,}")
    elif args.solver_type == solver_types.block_iterative:
        opts = {
                    'ksp_type': 'fgmres',
                    'pc_type': 'ilu',
                    'pc_factor_levels': 0,
                    'pc_factor_fill': 2.0,
                    }

        solver = scifem.NewtonSolver(F, J, [u, lmbda1, lmbda2, V_cell], bcs=[], petsc_options=opts)

        PETSc.Sys.Print(f"Solving problem with total current condition of {np.abs(I_tot.value):.3f} [A], n_dofs: {n_dofs:,}")

        t0 = time.time()
        solver.solve()
        t1 = time.time()
        PETSc.Sys.Print(f"Solve time: {t1 - t0:.3f}s, n_dofs: {n_dofs:,}")
    else:
        raise ValueError(f"Not implemented for solver type {args.solver_type}")

    current_l = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(inner(-kappa * grad(u), n)) * ds(markers.left))), op=MPI.SUM)
    current_r = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(inner(-kappa * grad(u), n)) * ds(markers.right))), op=MPI.SUM)
    current_ins = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(inner(-kappa * grad(u), n)) * ds(markers.insulated))), op=MPI.SUM)
    i_sup_right = current_r / L_right
    i_stdev_right = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((inner(kappa*grad(u), n)-i_sup_right) ** 2 * ds(markers.right))), op=MPI.SUM) / L_right)
    u_avg_right = comm.allreduce(fem.assemble_scalar(fem.form(u * ds(markers.right))), op=MPI.SUM) / L_right
    sd_right = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((u-u_avg_right) ** 2 * ds(markers.right))), op=MPI.SUM) / L_right)
    PETSc.Sys.Print(f"Current left boundary: {current_l:.3f} [A],", f"Current right boundary: {current_r:.3f} [A], ", f"Current insulated boundary: {current_ins:.3f} [A]")
    PETSc.Sys.Print(f"Avg potential right: {u_avg_right}, std potential right: {sd_right}")
    PETSc.Sys.Print(f"Avg i right: {i_sup_right}, std i right: {i_stdev_right}")
    PETSc.Sys.Print(f"L_left: {L_left:.1f}, L_bottom: {L_bottom:.1f}, L_right: {L_right:.1f}, L_top: {L_top:.1f}")

    with VTXWriter(comm, "potential.bp", [u], engine="BP5") as vtx:
        vtx.write(0.0)

    cells, types, x = plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = u.x.array.real
    grid.set_active_scalars("u")
    plotter = pv.Plotter()

    # plot potential heatmap
    plotter.add_mesh(grid, show_edges=False, opacity=0.25)

    # plot isopotential lines
    contour_levels = 10
    contours = grid.contour(contour_levels, scalars="u")
    plotter.add_mesh(contours, line_width=1)
    plotter.view_xy() # orientation
    plotter.show()
