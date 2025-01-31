
import gmsh
import numpy as np
import scifem
import ufl

from dolfinx import cpp, default_scalar_type, fem, mesh
from dolfinx.io import gmshio, VTXWriter
from mpi4py import MPI
from petsc4py import PETSc
from ufl import inner, grad

import solvers

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


def build_mesh(output_path, markers, Lx=10, Ly=1):
    """
    generate mesh for given dimensions (`Lx` `Ly`) and write output to `output_path`
    """
    gmsh.initialize()
    gmsh.model.add('2D')
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.01)
    coords = [
    (0, 0, 0),
    (Lx, 0, 0),
    (5*Lx, 0.5*Ly, 0),
    (3*Lx, Ly, 0),
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
    markers = Boundaries()
    output_mesh_path = 'mesh.msh'
    output_potential_path = 'potential.bp'
    build_mesh(output_mesh_path, markers, Lx=1)

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

    # facets submesh
    submesh_facets, submesh_facets_to_mesh, f_v_map = mesh.create_submesh(
        domain, fdim, ft.find(markers.right))[:3]
    num_facets_local = (
        domain.topology.index_map(fdim).size_local + domain.topology.index_map(fdim).num_ghosts
    )
    parent_to_facets = np.full(num_facets_local, -1, dtype=np.int32)
    parent_to_facets[submesh_facets_to_mesh] = np.arange(len(submesh_facets_to_mesh), dtype=np.int32)
    entity_maps = {submesh_facets: parent_to_facets}

    all_facets = compute_cell_boundary_facets(domain, ct, markers.domain)
    right_facets = compute_interface_cell_boundary_facets(domain, ct, ft, markers.domain, markers.right)
    left_facets = compute_interface_cell_boundary_facets(domain, ct, ft, markers.domain, markers.left)
    minus_right_facets = delete_numpy_rows(all_facets, right_facets)
    right_bndry_facets = np.array(right_facets).flatten()
    left_bndry_facets = np.array(left_facets).flatten()

    Vg = fem.functionspace(submesh_facets, ("Lagrange", 2))
    g = fem.Function(Vg)
    dg = ufl.TestFunction(Vg)

    # # Create the measure
    dx = ufl.Measure('dx', domain=domain, subdomain_data=ct, subdomain_id=markers.domain)
    ds_c = ufl.Measure("ds", subdomain_data=[(1, minus_right_facets.flatten()), (2, right_bndry_facets), (3, left_bndry_facets)], domain=domain)
    dx_f = ufl.Measure('dx', domain=submesh_facets)

    R = scifem.create_real_functionspace(submesh_facets)

    u_left = fem.Function(V)
    with u_left.x.petsc_vec.localForm() as u0_loc:
        u0_loc.set(0)

    left_dofs = fem.locate_dofs_topological(V, 1, left_boundary)
    left_bc = fem.dirichletbc(u_left, left_dofs)

    u, lmbda = fem.Function(V), fem.Function(R)
    du, dl = ufl.TestFunction(V), ufl.TestFunction(R)

    zero = fem.Constant(submesh_facets, default_scalar_type(0.0))

    kappa = fem.Constant(domain, default_scalar_type(0.1))
    I_tot = fem.Constant(submesh_facets, PETSc.ScalarType(-1.0))
    maps = [(Wi.dofmap.index_map, Wi.dofmap.index_map_bs) for Wi in [V, R, Vg]]
    L_right = comm.allreduce(fem.assemble_scalar(fem.form(1 * ds(markers.right))), op=MPI.SUM)

    gamma = 5
    h = ufl.CellDiameter(domain)
    F0 = kappa * inner(grad(u), grad(du)) * dx - g * du * ds_c(2)
    # Left Dirichlet bc - Nitsche's method
    # F0 += - kappa * (u - u_left) * inner(n, grad(du)) * ds_c(3)
    # F0 += -gamma / h * (u - u_left) * du * ds_c(3)
    F1 = dl * g * ds_c(2) + I_tot/L_right * dl * ds_c(2)
    F2 = - dg * u * ds_c(2) + lmbda * dg * ds_c(2)

    F = [fem.form(F0, entity_maps=entity_maps), fem.form(F1, entity_maps=entity_maps), fem.form(F2, entity_maps=entity_maps)]
    j00 = ufl.derivative(F0, u)
    j01 = ufl.derivative(F0, lmbda)
    j02 = ufl.derivative(F0, g)
    j10 = ufl.derivative(F1, u)
    j11 = ufl.derivative(F1, lmbda)
    j12 = ufl.derivative(F1, g)

    j20 = ufl.derivative(F2, u)
    j21 = ufl.derivative(F2, lmbda)
    j22 = ufl.derivative(F2, g)

    J00 = fem.form(j00, entity_maps=entity_maps)
    J01 = fem.form(j01, entity_maps=entity_maps)
    J02 = fem.form(j02, entity_maps=entity_maps)

    J10 = fem.form(j10, entity_maps=entity_maps)
    J11 = fem.form(j11, entity_maps=entity_maps)
    J12 = fem.form(j12, entity_maps=entity_maps)

    J20 = fem.form(j20, entity_maps=entity_maps)
    J21 = fem.form(j21, entity_maps=entity_maps)
    J22 = fem.form(j22, entity_maps=entity_maps)

    J = [[J00, J01, J02], [J10, J11, J12], [J20, J21, J22]]

    opts = {
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                'pc_factor_mat_solver_type': 'superlu_dist',
                }
    solver = scifem.NewtonSolver(F, J, [u, lmbda, g], bcs=[left_bc], petsc_options=opts)

    PETSc.Sys.Print(f"Solving problem with total current condition of {np.abs(I_tot.value):.3f} [A]")
    solver.solve()

    current_l = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(inner(-kappa * grad(u), n)) * ds(markers.left))), op=MPI.SUM)
    current_r = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(inner(-kappa * grad(u), n)) * ds(markers.right))), op=MPI.SUM)
    current_ins = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(inner(-kappa * grad(u), n)) * ds(markers.insulated))), op=MPI.SUM)
    i_sup_right = current_r / L_right
    i_stdev_right = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((inner(kappa*grad(u), n)-i_sup_right) ** 2 * ds(markers.right))), op=MPI.SUM) / L_right)
    u_avg_right = comm.allreduce(fem.assemble_scalar(fem.form(u * ds(markers.right))), op=MPI.SUM) / L_right
    sd_right = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((u-u_avg_right) ** 2 * ds(markers.right))), op=MPI.SUM) / L_right)
    print(f"Current left boundary: {current_l:.3f} [A],", f"Current right boundary: {current_r:.3f} [A], ", f"Current insulated boundary: {current_ins:.3f} [A]")
    print(f"Avg potential right: {u_avg_right}, std potential right: {sd_right}")
    print(f"Avg i right: {i_sup_right}, std i right: {i_stdev_right}")

    with VTXWriter(comm, "potential.bp", [u], engine="BP5") as vtx:
        vtx.write(0.0)
