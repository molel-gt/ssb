#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import time

sys.path.append("../")

import gmsh
import matspy
import numpy as np
import scifem
import scipy
import ufl

from dolfinx import cpp, fem, io, mesh
from dolfinx.graph import partitioner_scotch
import dolfinx.fem.petsc as petsc

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from ufl import div, dot, grad, inner
import cffi
import numba
from ffcx.codegeneration.utils import get_void_pointer

import commons, mesh_utils, solvers, utils

markers = commons.Markers()
Print = PETSc.Sys.Print
dtype = PETSc.ScalarType
_type_to_offset_index = {fem.IntegralType.cell: 0, fem.IntegralType.exterior_facet: 1}

ffi = cffi.FFI()


def create_mesh(LX, LY):
    gmsh.initialize()
    gmsh.model.add('ssb')
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.01)

    points = [
        (0, 0, 0),
        (0.5*LX, 0, 0),
        (LX, 0, 0),
        (LX, LY, 0),
        (0.5*LX, LY, 0),
        (0, LY, 0),
        ]
    gpoints = [gmsh.model.occ.addPoint(*p) for p in points]
    lines = [gmsh.model.occ.addLine(gpoints[i], gpoints[i+1]) for i in range(-1, len(gpoints)-1)]
    lines.append(gmsh.model.occ.addLine(gpoints[1], gpoints[4]))

    se_loop = gmsh.model.occ.addCurveLoop([1, 2, 7, 6])
    am_loop = gmsh.model.occ.addCurveLoop([3, 4, 5, 7])
    se_surf = gmsh.model.occ.addPlaneSurface([se_loop])
    am_surf = gmsh.model.occ.addPlaneSurface([am_loop])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(1, [1], markers.left, "Left")
    gmsh.model.addPhysicalGroup(1, [4], markers.right, "Right")
    gmsh.model.addPhysicalGroup(1, [7], markers.electrolyte_v_positive_am, "SE/AM")

    gmsh.model.addPhysicalGroup(2, [se_surf], markers.electrolyte, "SE")
    gmsh.model.addPhysicalGroup(2, [am_surf], markers.positive_am, "AM")
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write("mesh.msh")
    gmsh.finalize()


def facets_for_subdomain(msh, ft, ct_marker):
    tdim = msh.topology.dim
    fdim = tdim - 1
    f_to_c = msh.topology.connectivity(fdim, tdim)
    c_to_f = msh.topology.connectivity(tdim, fdim)
    facets = []
    for f in ft.indices:
        if len(f_to_c.links(f)) == 2:
            c_0, c_1 = f_to_c.links(f)[0], f_to_c.links(f)[1]
            subdomain_0, subdomain_1 = ct.values[[c_0, c_1]]
            if subdomain_0 == ct_marker:
                facets.append(f)
            elif subdomain_1 == ct_marker:
                facets.append(f)
        elif len(f_to_c.links(f)) == 1:
            c_0 = f_to_c.links(f)[0]
            subdomain_0, = ct.values[[c_0]]
            if subdomain_0 == ct_marker:
                facets.append(f)

    return np.array(facets)


def tagged_cell_boundary_facets(msh, ct, ft):
    tdim = msh.topology.dim
    fdim = tdim - 1
    f_to_c = msh.topology.connectivity(fdim, tdim)
    c_to_f = msh.topology.connectivity(tdim, fdim)
    internal_se = []
    internal_am = []
    se_xface = []
    am_xface = []
    ft_imap = msh.topology.index_map(fdim)
    for c_id in ct.indices:
        facets = c_to_f.links(c_id)
        subdomain = ct.values[c_id]
        for f in facets:
            if f >= ft_imap.size_local:
                continue
            ft_id = ft.values[[f]]
            if ft_id == markers.electrolyte_v_positive_am:
                local_f = np.where(c_to_f.links(c_id) == f)[0][0]
                if subdomain == markers.electrolyte:
                    se_xface.extend((c_id, local_f))
                else:
                    am_xface.extend((c_id, local_f))
            else:
                local_f = np.where(c_to_f.links(c_id) == f)[0][0]
                if subdomain == markers.electrolyte:
                    internal_se.extend((c_id, local_f))
                elif subdomain == markers.positive_am:
                    internal_am.extend((c_id, local_f))
    
    return np.array(internal_se), np.array(internal_am), np.array(se_xface), np.array(am_xface)


class LocalAssembler():

    def __init__(self, form, entity_maps, integral_type: fem.IntegralType =fem.IntegralType.cell):
        self.form = fem.form(form, entity_maps=entity_maps)
        self.integral_type = integral_type
        self.update_coefficients()
        self.update_constants()

        is_complex = np.issubdtype(ScalarType, np.complexfloating)
        nptype = "complex128" if is_complex else "float64"
        o_s = self.form.ufcx_form.form_integral_offsets[_type_to_offset_index[integral_type]]
        o_e = self.form.ufcx_form.form_integral_offsets[_type_to_offset_index[integral_type]+1]
        assert o_e - o_s == 1

        self.kernel = getattr(self.form.ufcx_form.form_integrals[o_s], f"tabulate_tensor_{nptype}")
        self.active_cells = self.form._cpp_object.domains(integral_type, 0)
        # assert len(self.form.function_spaces) == 2
        self.local_shape = [0,0]
        for i, V in enumerate(self.form.function_spaces):
            self.local_shape[i] = V.dofmap.dof_layout.block_size * V.dofmap.dof_layout.num_dofs

        # e0 = self.form.function_spaces[0].element
        # # e1 = self.form.function_spaces[1].element
        # needs_transformation_data = e0.needs_dof_transformations or \
        #     self.form._cpp_object.needs_facet_permutations
        # if needs_transformation_data:
        #     raise NotImplementedError("Dof transformations not implemented")

    def update_coefficients(self):
        self.coeffs = fem.assemble.pack_coefficients(self.form)

    def update_constants(self):
        self.consts = fem.assemble.pack_constants(self.form)

    def update(self):
        self.update_coefficients()
        self.update_constants()


@numba.njit(fastmath=True)
def assemble_matrix(kernel, mesh, local_shape, cell_idx):
    x_dofs, x = mesh
    geometry = np.zeros((len(x_dofs[cell_idx]), 3), dtype=x.dtype)
    geometry[:, :] = x[x_dofs[cell_idx]]

    A_local = np.zeros((local_shape[0], local_shape[1]), dtype=ScalarType)
    facet_index = np.array([0], dtype=np.intc)
    facet_perm = np.zeros(0, dtype=np.uint8)
    coeffs = np.zeros(0, dtype=ScalarType)
    consts = np.zeros(1, dtype=ScalarType)
    custom_data = np.zeros(1, dtype=np.int64)
    custom_data_ptr = get_void_pointer(custom_data)
    ffi_fb = ffi.from_buffer
    kernel(ffi_fb(A_local), ffi_fb(coeffs), ffi_fb(consts), ffi_fb(geometry),
           ffi_fb(facet_index), ffi_fb(facet_perm), custom_data_ptr)
    return A_local


@numba.njit(fastmath=True)
def assemble_vector(kernel, mesh, local_shape, cell_idx, coeffs):
    x_dofs, x = mesh
    geometry = np.zeros((len(x_dofs[cell_idx]), 3), dtype=x.dtype)
    geometry[:, :] = x[x_dofs[cell_idx]]

    b_local = np.zeros((local_shape[0],), dtype=ScalarType)
    facet_index = np.array([0], dtype=np.intc)
    facet_perm = np.zeros(0, dtype=np.uint8)
    l_coeffs = coeffs[cell_idx, :]
    consts = np.zeros(1, dtype=ScalarType)
    custom_data = np.zeros(1, dtype=np.int64)
    custom_data_ptr = get_void_pointer(custom_data)
    ffi_fb = ffi.from_buffer
    kernel(ffi_fb(b_local), ffi_fb(coeffs), ffi_fb(consts), ffi_fb(geometry),
           ffi_fb(facet_index), ffi_fb(facet_perm), custom_data_ptr)
    return b_local


if __name__ == '__main__':
    # create_mesh(1, 0.5)
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument("-m", "--mesh_folder", help="folder containing mesh files", required=True)
    parser.add_argument("-p_u0", '--poly_order_u0', help='polynomial approximation order for u0',  nargs='?', type=int, const=1, default=1)
    parser.add_argument("-p_u1", '--poly_order_u1', help='polynomial approximation order for u1',  nargs='?', type=int, const=1, default=1)
    parser.add_argument("-kr", '--kr', help='ionic to electronic conductivity ratio',  nargs='?', type=float, const=1, default=1.0)
    parser.add_argument("-k", '--k', help='polynomial approximation order',  nargs='?', type=int, const=1, default=1)
    args = parser.parse_args()
    output_meshfile = os.path.join(args.mesh_folder, "mesh.msh")
    results_folder = os.path.join(args.mesh_folder, f"output/k_{args.k}/kr_{args.kr}")
    utils.make_dir_if_missing(results_folder)
    comm = MPI.COMM_WORLD
    partitioner = mesh.create_cell_partitioner(partitioner_scotch(), mesh.GhostMode.shared_facet)
    domain, ct, ft = io.gmsh.read_from_msh(output_meshfile, comm, partitioner=partitioner)[:3]
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_entities(fdim)
    domain.topology.create_connectivity(tdim, fdim)
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(fdim, fdim)
    ct_imap = domain.topology.index_map(tdim)
    num_entities_local = ct_imap.size_local + ct_imap.num_ghosts
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

    tagged_boundary_facets = tagged_cell_boundary_facets(domain, ct, ft)

    submesh_electrolyte, se_mesh_emap = mesh.create_submesh(
        domain, tdim, ct.find(markers.electrolyte)
    )[:2]
    submesh_positive_am, am_mesh_emap = mesh.create_submesh(
        domain, tdim, ct.find(markers.positive_am)
    )[:2]

    domain.topology.create_connectivity(fdim, tdim)
    f_to_c = domain.topology.connectivity(fdim, tdim)

    x = ufl.SpatialCoordinate(domain)
    dx = ufl.Measure('dx', domain=domain, subdomain_data=ct)

    # facet mesh
    facet_mesh, facet_mesh_emap = mesh.create_submesh(domain, fdim, ft.indices)[:2]
    se_ft = facets_for_subdomain(domain, ft, markers.electrolyte)
    am_ft = facets_for_subdomain(domain, ft, markers.positive_am)
    se_ft_mesh, se_ft_mesh_emap = mesh.create_submesh(domain, fdim, se_ft)[:2]
    am_ft_mesh, am_ft_mesh_emap = mesh.create_submesh(domain, fdim, am_ft)[:2]
    entity_maps = [se_mesh_emap, am_mesh_emap, se_ft_mesh_emap, am_ft_mesh_emap]

    V = fem.functionspace(domain, ("DG", args.k))
    Vbar = fem.functionspace(facet_mesh, ("DG", args.k))

    V0 = fem.functionspace(submesh_electrolyte, ("DG", args.k))
    V0bar = fem.functionspace(se_ft_mesh, ("DG", args.k))
    V1 = fem.functionspace(submesh_positive_am, ("DG", args.k))
    V1bar = fem.functionspace(am_ft_mesh, ("DG", args.k))

    u, ubar = fem.Function(V), fem.Function(Vbar)
    v, vbar = ufl.TestFunction(V), ufl.TestFunction(Vbar)

    u0, v0 = fem.Function(V0), ufl.TestFunction(V0)
    u0bar, v0bar = fem.Function(V0bar), ufl.TestFunction(V0bar)
    u1, v1 = fem.Function(V1), ufl.TestFunction(V1)
    u1bar, v1bar = fem.Function(V1bar), ufl.TestFunction(V1bar)
    du0, du1 = ufl.TrialFunction(V0), ufl.TrialFunction(V1)
    du0bar, du1bar = ufl.TrialFunction(V0bar), ufl.TrialFunction(V1bar)

    h = ufl.CellDiameter(domain)
    n = ufl.FacetNormal(domain)

    h0 = ufl.CellDiameter(submesh_electrolyte)
    n0 = ufl.FacetNormal(submesh_electrolyte)

    h1 = ufl.CellDiameter(submesh_positive_am)
    n1 = ufl.FacetNormal(submesh_positive_am)

    gamma = 16.0 * args.k**2 / h
    gamma0 = 2.0/3.0 * args.k**2 / h0
    gamma1 = 2.0/3.0 * args.k**2 / h1

    # Create the measure
    ds_c = ufl.Measure("ds", subdomain_data=[(markers.electrolyte, tagged_boundary_facets[0]),
                       (markers.positive_am, tagged_boundary_facets[1]),
                       (markers.electrolyte_v_positive_am, tagged_boundary_facets[2]),
                       (markers.electrolyte_v_positive_am*11, tagged_boundary_facets[3])], domain=domain)

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
    dInterface = ufl.Measure("dS", domain=domain, subdomain_data=int_facet_domains, subdomain_id=markers.electrolyte_v_positive_am)
    ds = ufl.Measure('ds', domain=domain, subdomain_data=ft)

    FaradayConstant = 96485
    R = 8.314
    T = 298
    i0 = 1.0e2
    sigma = 0.1
    kappa = args.kr * sigma
    eta_s = -R * T / i0 / FaradayConstant * inner(sigma * grad(u1("+")), n1("+"))
    U_ocv = 0.7
    u_l = u1("+") - U_ocv - eta_s
    F0 = kappa * inner(grad(u0), grad(v0)) * dx(markers.electrolyte)
    F0 += - kappa * inner(grad(u0), n0) * v0 * (ds_c(markers.electrolyte) + ds_c(markers.electrolyte_v_positive_am))
    F0 += + kappa * (u0 - u0bar) * inner(grad(v0), n0) * ds_c(markers.electrolyte)
    F0 += + gamma0 * (u0 - u0bar) * v0 * ds_c(markers.electrolyte)
    F0 += + kappa * (u0("-") - u_l) * inner(grad(v0("-")), n0("-")) * dInterface
    F0 += + gamma0("-") * (u0("-") - u_l) * v0("-") * dInterface

    F0_bar = kappa * inner(grad(u0), n0) * v0bar * (ds_c(markers.electrolyte) + ds_c(markers.electrolyte_v_positive_am))
    F0_bar += - gamma0 * (u0 - u0bar) * v0bar * (ds_c(markers.electrolyte) + ds_c(markers.electrolyte_v_positive_am))
    # F0_bar += + gamma0("-") * (u0("-") - u_l) * v0bar("-") * dInterface

    F1 = sigma * inner(grad(u1), grad(v1)) * dx(markers.positive_am) 
    F1 += - sigma * inner(grad(u1), n1) * v1 * (ds_c(markers.positive_am) + ds_c(markers.electrolyte_v_positive_am*11))
    F1 += + sigma * (u1 - u1bar) * inner(grad(v1), n1) * (ds_c(markers.positive_am) + ds_c(markers.electrolyte_v_positive_am*11))
    F1 += + gamma1 * (u1 - u1bar) * v1 * (ds_c(markers.positive_am) + ds_c(markers.electrolyte_v_positive_am*11))

    F1_bar = sigma * inner(grad(u1), n1) * v1bar * (ds_c(markers.positive_am) + ds_c(markers.electrolyte_v_positive_am*11))
    F1_bar += - gamma1 * (u1 - u1bar) * v1bar * (ds_c(markers.positive_am) + ds_c(markers.electrolyte_v_positive_am*11))
    F1_bar += -kappa * inner(grad(u0("-")), n1("+")) * v1bar("+") * dInterface

    j00 = ufl.derivative(F0, u0, du0)
    j01 = ufl.derivative(F0, u0bar, du0bar)
    j02 = ufl.derivative(F0, u1, du1)
    j03 = ufl.derivative(F0, u1bar, du1bar)

    j10 = ufl.derivative(F0_bar, u0, du0)
    j11 = ufl.derivative(F0_bar, u0bar, du0bar)
    j12 = ufl.derivative(F0_bar, u1, du1)
    j13 = ufl.derivative(F0_bar, u1bar, du1bar)

    j20 = ufl.derivative(F1, u0, du0)
    j21 = ufl.derivative(F1, u0bar, du0bar)
    j22 = ufl.derivative(F1, u1, du1)
    j23 = ufl.derivative(F1, u1bar, du1bar)

    j30 = ufl.derivative(F1_bar, u0, du0)
    j31 = ufl.derivative(F1_bar, u0bar, du0bar)
    j32 = ufl.derivative(F1_bar, u1, du1)
    j33 = ufl.derivative(F1_bar, u1bar, du1bar)

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

    F = [
        fem.form(F0, entity_maps=entity_maps),
        fem.form(F0_bar, entity_maps=entity_maps),
        fem.form(F1, entity_maps=entity_maps),
        fem.form(F1_bar, entity_maps=entity_maps),
    ]

    # plot sparsity
    J_view = fem.petsc.assemble_matrix(J, kind="mpi")
    J_view.assemble()
    ai, aj, av = J_view.getValuesCSR()
    Asp = scipy.sparse.csr_matrix((av, aj, ai))
    fig, ax = matspy.spy_to_mpl(Asp)
    ax.set_title('')
    fig.savefig("jacobian-sparsity.eps", bbox_inches='tight')

    J2D = fem.form(J)
    F2D = fem.form(F)
    Jmat2d = fem.petsc.create_matrix(J2D)
    Fvec2d = fem.petsc.create_vector([V0, V0bar, V1, V1bar], kind="mpi")
    snes = PETSc.SNES().create(comm)
    snes.setType('newtonls')
    snes.setTolerances(rtol=1e-7, max_it=200)
    snes.setMonitor(lambda _, it, residual: Print("it:", it, "res:", residual))
    snes.getKSP().setType(PETSc.KSP.Type.CG)
    snes.getKSP().getPC().setType(PETSc.PC.Type.LU)
    # snes.getKSP().getPC().setFactorSolverType("mumps")
    snes.getKSP().setOptionsPrefix("snes_")
    snes.getKSP().setOperators(Jmat2d, Jmat2d)
    snes.getKSP().setTolerances(rtol=1e-7)
    snes.setErrorIfNotConverged(True)
    snes.getKSP().setErrorIfNotConverged(True)
    snes.getKSP().setConvergenceHistory()

    # Since the boundary condition is enforced in the facet space, we need
    # to get the corresponding facets in `facet_mesh` using the entity map
    se_ft_mesh.topology.create_connectivity(fdim, fdim)
    am_ft_mesh.topology.create_connectivity(fdim, fdim)
    left_boundary_facets = se_ft_mesh_emap.sub_topology_to_topology(
        ft.find(markers.left), inverse=True
    )
    right_boundary_facets = am_ft_mesh_emap.sub_topology_to_topology(
        ft.find(markers.right), inverse=True
    )

    # Get the dofs and apply the boundary condition
    facet_mesh.topology.create_connectivity(fdim, fdim)
    left_dofs = fem.locate_dofs_topological(V0bar, fdim, left_boundary_facets)
    right_dofs = fem.locate_dofs_topological(V1bar, fdim, right_boundary_facets)
    left_bc = fem.dirichletbc(dtype(0.0), left_dofs, V0bar)
    right_bc = fem.dirichletbc(dtype(1.0), right_dofs, V1bar)
    bcs = [left_bc, right_bc]
    problem_t0 = solvers.NonlinearPDE_SNESProblem(F2D, J2D, [u0, u0bar, u1, u1bar], bcs, P=J2D)
    snes.setFunction(problem_t0.F_block, Fvec2d)
    snes.setJacobian(problem_t0.J_block, J=Jmat2d, P=Jmat2d)
    x2d = fem.petsc.create_vector([V0, V0bar, V1, V1bar], kind="mpi")
    x2d.set(0.0)
    V0_map = V0.dofmap.index_map
    V0_dofmap = V0.dofmap
    V1_map = V1.dofmap.index_map
    V1_dofmap = V1.dofmap
    V0bar_map = V0bar.dofmap.index_map
    V0bar_dofmap = V0bar.dofmap
    V1bar_map = V1bar.dofmap.index_map
    V1bar_dofmap = V1bar.dofmap
    n_dofs = V0_map.size_global*V0.dofmap.index_map_bs + V1_map.size_global*V1.dofmap.index_map_bs +\
        V0bar_map.size_global*V0bar.dofmap.index_map_bs + V1bar_map.size_global*V1bar.dofmap.index_map_bs
    Print(f"Solving problem, dofs: {n_dofs:,}")
    t0 = time.time()
    snes.solve(None, x2d)
    t1 = time.time()
    Print(f"Solved problem in {t1-t0:,.3f} seconds")
    snes.destroy()
    Jmat2d.destroy()
    Fvec2d.destroy()
    x2d.destroy()
    i_left = comm.allreduce(fem.assemble_scalar(fem.form(inner(kappa * grad(u0), n0) * ds(markers.left), entity_maps=entity_maps)), op=MPI.SUM)
    i_x_left = comm.allreduce(fem.assemble_scalar(fem.form(inner(kappa * grad(u0)("-"), n0("-")) * dInterface, entity_maps=entity_maps)), op=MPI.SUM)
    i_x_right = comm.allreduce(fem.assemble_scalar(fem.form(inner(sigma * grad(u1)("+"), n1("+")) * dInterface, entity_maps=entity_maps)), op=MPI.SUM)
    i_right = comm.allreduce(fem.assemble_scalar(fem.form(inner(sigma * grad(u1), n1) * ds(markers.right), entity_maps=entity_maps)), op=MPI.SUM)
    Print(i_left, i_x_left, i_x_right, i_right)
    error = kappa * inner(grad(u0)('-'), n0("-")) + sigma * inner(grad(u1)('+'), n1("+"))
    i_x_error = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(inner(error, error) * dInterface, entity_maps=entity_maps)), op=MPI.SUM))
    i_x_norm_l = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(inner(inner(kappa * grad(u0)("-"), n0("-")), inner(kappa * grad(u0)("-"), n0("-"))) * dInterface, entity_maps=entity_maps)), op=MPI.SUM))
    i_x_norm_r = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(inner(inner(sigma * grad(u1)("+"), n1("+")), inner(sigma * grad(u1)("+"), n1("+"))) * dInterface, entity_maps=entity_maps)), op=MPI.SUM))
    error_norm = i_x_norm_l / i_x_norm_r
    error_norm_2 = i_x_error / (0.5 * i_x_norm_l + 0.5 * i_x_norm_r)
    Print(i_x_norm_l, i_x_norm_r)
    Print(f'error 1: {error_norm}, error 2: {error_norm_2}')

    se_cell_imap = submesh_electrolyte.topology.index_map(tdim)
    se_cells = np.arange(se_cell_imap.size_local + se_cell_imap.num_ghosts)
    parent_cells = se_mesh_emap.sub_topology_to_topology(se_cells, inverse=False)
    u.interpolate(u0, cells1=parent_cells, cells0=se_cells)
    am_cell_imap = submesh_positive_am.topology.index_map(tdim)
    am_cells = np.arange(am_cell_imap.size_local + am_cell_imap.num_ghosts)
    parent_cells = am_mesh_emap.sub_topology_to_topology(am_cells, inverse=False)
    u.interpolate(u1, cells1=parent_cells, cells0=am_cells)
    se_ft = np.arange(se_ft_mesh.topology.index_map(fdim).size_local + se_ft_mesh.topology.index_map(fdim).num_ghosts)
    se_ft_parent = se_ft_mesh_emap.sub_topology_to_topology(se_ft, inverse=False)
    am_ft = np.arange(am_ft_mesh.topology.index_map(fdim).size_local + am_ft_mesh.topology.index_map(fdim).num_ghosts)
    am_ft_parent = am_ft_mesh_emap.sub_topology_to_topology(am_ft, inverse=False)
    ubar.interpolate(u0bar, cells1=se_ft_parent, cells0=se_ft)
    ubar.interpolate(u1bar, cells1=am_ft_parent, cells0=am_ft)

    with io.VTXWriter(domain.comm, os.path.join(results_folder, "u.bp"), [u], "bp5") as f:
        f.write(0.0)
    with io.VTXWriter(domain.comm, os.path.join(results_folder, "ubar.bp"), [ubar], "bp5") as f:
        f.write(0.0)

    shutil.rmtree(os.path.join(os.environ["HOME"], ".cache/fenics"), ignore_errors=True)

    R_fun = scifem.create_real_functionspace(domain)
    Vp1 = fem.functionspace(domain, ("DG", args.k+1))
    W = ufl.MixedFunctionSpace(Vp1, R_fun)
    q, lmda = ufl.TrialFunctions(W)
    dq, dlmda = ufl.TestFunctions(W)
    wh = fem.Function(Vp1)
    kappa_a = 0.1
    cells = np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32)
    zero = fem.Constant(domain, dtype(0.0))

    a00 = inner(kappa_a * grad(q), grad(dq)) * dx
    a01 = lmda * dq * dx
    a10 = q * dlmda * dx
    a11 = None
    L0 = - inner(div(kappa_a * grad(u)), dq) * dx - kappa_a * inner(grad(u), n) * dq * ds_c
    L1 = inner(zero, dlmda) * dx
    a_form = fem.form([
        [a00, a01],
        [a10, None]
        ])
    L_form = fem.form([L0, L1])
    x_dofs = V.mesh.geometry.dofmap
    x = domain.geometry.x
    bs = Vp1.dofmap.bs
    a00_assembler = LocalAssembler(a00, entity_maps)
    a00_kernel = a00_assembler.kernel
    a00_lshape = a00_assembler.local_shape
    a01_assembler = LocalAssembler(a01, entity_maps)
    a01_kernel = a01_assembler.kernel
    a01_lshape = a01_assembler.local_shape
    a10_assembler = LocalAssembler(a10, entity_maps)
    a10_kernel = a10_assembler.kernel
    a10_lshape = a10_assembler.local_shape
    b0_assembler = LocalAssembler(L0, entity_maps)
    b0_kernel = b0_assembler.kernel
    b0_lshape = b0_assembler.local_shape
    b1_assembler = LocalAssembler(L1, entity_maps)
    b1_kernel = b1_assembler.kernel
    b1_lshape = b1_assembler.local_shape

    ct_u = mesh.meshtags(domain, tdim, ct.indices, ct.indices)
    dx_u = ufl.Measure('dx', domain=domain, subdomain_data=ct_u)
    u_tot = lambda v: fem.assemble_scalar(fem.form(u * dx_u(v)))
    imap = domain.topology.index_map(tdim)

    for cell_idx in range(imap.size_local):
        idx_global = imap.local_to_global(np.array([cell_idx], dtype=np.int32))[0]
        A00 = assemble_matrix(a00_kernel, (x_dofs, x), a00_lshape, cell_idx)
        A01 = assemble_matrix(a01_kernel, (x_dofs, x), a01_lshape, cell_idx)
        A10 = assemble_matrix(a10_kernel, (x_dofs, x), a10_lshape, cell_idx)
        A = np.zeros((A00.shape[0]+A10.shape[0], A00.shape[1]+A01.shape[1] ))

        A[:A00.shape[0], :A00.shape[0]] = A00
        A[:A00.shape[0], -1] = A01.T
        A[-1, :A00.shape[0]] = A10[:]
        b0 = assemble_vector(b0_kernel, (x_dofs, x), b0_lshape, cell_idx, b0_assembler.coeffs[(fem.IntegralType.cell, 0)])
        b1 = assemble_vector(b1_kernel, (x_dofs, x), b1_lshape, cell_idx, b1_assembler.coeffs[(fem.IntegralType.cell, 0)])
        b = np.zeros((A.shape[0],))
        b[:A00.shape[0]] = b0
        b[-1] = u_tot(idx_global)

        cell_dofs = Vp1.dofmap.cell_dofs(cell_idx)
        unrolled_dofs = np.zeros(len(cell_dofs)*bs, dtype=np.int32)
        for i, dof in enumerate(cell_dofs):
            for j in range(bs):
                unrolled_dofs[i*bs+j] = dof*bs+j
        wh.x.array[unrolled_dofs] = np.linalg.solve(A, b)[:A00.shape[0]]
    wh.x.scatter_forward()

    with io.VTXWriter(domain.comm, os.path.join(results_folder, "w.bp"), [wh], "bp5") as f:
        f.write(0.0)
