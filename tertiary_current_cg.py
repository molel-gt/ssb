# SPDX-License-Identifier: MIT
import argparse
import os
import timeit


import dolfinx

import dolfinx.fem.petsc
import ufl
import numpy as np

from dolfinx import cpp, fem, io, mesh
from mpi4py import MPI
from petsc4py import PETSc
from ufl import dot, grad, inner

import commons, constants, mesh_utils, solvers, utils 


R = 8.314
T = 298
faraday_const = 96485
kappa_pos_am = 0.1
kinetics = ('linear', 'tafel', 'butler_volmer')
micron = 1e-6


def define_interior_eq(domain, degree,  submesh, submesh_to_mesh, value, kappa):
    # Compute map from parent entity to submesh cell
    codim = domain.topology.dim - submesh.topology.dim
    ptdim = domain.topology.dim - codim
    num_entities = (
        domain.topology.index_map(ptdim).size_local
        + domain.topology.index_map(ptdim).num_ghosts
    )
    mesh_to_submesh = np.full(num_entities, -1)
    mesh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh), dtype=np.int32)

    V = fem.functionspace(submesh, ("Lagrange", degree))
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    ct_r = mesh.meshtags(domain, domain.topology.dim, submesh_to_mesh, np.full_like(submesh_to_mesh, 1, dtype=np.int32))
    val = fem.Constant(submesh, value)
    dx_r = ufl.Measure("dx", domain=domain, subdomain_data=ct_r, subdomain_id=1)
    F = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_r - val * v * dx_r
    return u, F, mesh_to_submesh


def mixed_term(u, v, n):
    return ufl.dot(ufl.grad(u), n) * v


def surface_overpotential(kappa, u, n, i0, kinetics_type='linear'):
    i_loc = -inner((kappa * grad(u)), n)
    if kinetics_type == "butler_volmer":
        return 2 * ufl.ln(0.5 * i_loc/i0 + ufl.sqrt((0.5 * i_loc/i0)**2 + 1)) * (R * T / faraday_const)
    elif kinetics_type == "linear":
        return R * T * i_loc / (i0 * faraday_const)
    elif kinetics_type == "tafel":
        return ufl.sign(i_loc) * R * T / (0.5 * faraday_const) * ufl.ln(np.abs(i_loc)/i_0)


def arctanh(y):
    return 0.5 * ufl.ln((1 + y) / (1 - y))


def ocv(c, cmax=35000):
    xi = 2 * (c - 0.5 * cmax) / cmax
    return 3.25 - 0.5 * arctanh(xi)


def ocv_simple(c, cmax=35000):
    return 4.2 * (1 - c/cmax) ** 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='secondary current distribution')
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1.0, type=float)
    parser.add_argument("--u_ocv", help="open-circuit potential", nargs='?', const=1, default=0, type=float)
    parser.add_argument("--Wa_n", help="Wagna number for negative electrode: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e-3, type=float)
    parser.add_argument("--Wa_p", help="Wagna number for positive electrode: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e3, type=float)
    parser.add_argument("--kr", help="ratio of ionic to electronic conductivity", nargs='?', const=1, default=1, type=float)
    parser.add_argument("--gamma", help="interior penalty parameter", nargs='?', const=1, default=15, type=float)
    parser.add_argument("--atol", help="solver absolute tolerance", nargs='?', const=1, default=1e-12, type=float)
    parser.add_argument("--rtol", help="solver relative tolerance", nargs='?', const=1, default=1e-9, type=float)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='MICRON_TO_METER', type=str)
    parser.add_argument('--kinetics', help='kinetics type', nargs='?', const=1, default='butler_volmer', type=str, choices=kinetics)
    parser.add_argument("--plot", help="whether to plot results", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    start_time = timeit.default_timer()
    voltage = args.voltage
    Wa_n = args.Wa_n
    Wa_p = args.Wa_p
    gamma = args.gamma
    kappa_elec = args.kr * kappa_pos_am
    dt = 1e-3
    D = 1e-15

    markers = commons.Markers()
    comm = MPI.COMM_WORLD

    dimensions = utils.extract_dimensions_from_meshfolder(args.mesh_folder)
    LX, LY, LZ = [float(vv) * micron for vv in dimensions.split("-")]

    characteristic_length = LZ 
    if np.isclose(LZ, 0):
        characteristic_length = LX

    output_meshfile = os.path.join(args.mesh_folder, "mesh.msh")
    results_dir = os.path.join(args.mesh_folder, args.kinetics, str(Wa_n) + "-" + str(Wa_p) + "-" + str(args.kr), str(args.gamma))
    utils.make_dir_if_missing(results_dir)
    output_potential_file = os.path.join(results_dir, "potential.bp")
    elec_potential_file = os.path.join(results_dir, "electrolyte_potential.bp")
    positive_am_potential_file = os.path.join(results_dir, "positive_am_potential.bp")
    current_file = os.path.join(results_dir, "current.bp")
    concentration_file = os.path.join(results_dir, "concentration.bp")

    # load mesh
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = io.gmshio.read_from_msh(output_meshfile, comm, partitioner=partitioner)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)
    domain.topology.create_connectivity(tdim, tdim)
    domain.topology.create_connectivity(fdim, fdim)

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
    # entity_maps = {submesh_electrolyte._cpp_object: parent_to_sub_electrolyte, submesh_positive_am._cpp_object: parent_to_sub_positive_am}


    u_0, F_00, m_to_elec = define_interior_eq(domain, 2, submesh_electrolyte, submesh_electrolyte_to_mesh, 0.0, kappa_elec)
    u_1, F_11, m_to_pos_am = define_interior_eq(domain, 2, submesh_positive_am, submesh_positive_am_to_mesh, 0.0, kappa_pos_am)
    u_0.name = "u_b"
    u_1.name = "u_t"


    # Add coupling term to the interface
    # Get interface markers on submesh b
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
    # dInterface = ufl.Measure("dS", domain=domain, subdomain_data=ft, subdomain_id=markers.electrolyte_v_positive_am)
    dx_r = ufl.Measure('dx', domain=domain, subdomain_data=ct, subdomain_id=markers.positive_am)
    ds = ufl.Measure('ds', domain=domain, subdomain_data=ft)
    ds_r = ufl.Measure('ds', domain=submesh_positive_am, subdomain_data=ft_positive_am)
    l_res = "+"
    r_res = "-"

    v_l = ufl.TestFunction(u_0.function_space)(l_res)
    v_r = ufl.TestFunction(u_1.function_space)(r_res)
    u_l = u_0(l_res)
    u_r = u_1(r_res)


    n = ufl.FacetNormal(domain)
    n2 = ufl.FacetNormal(submesh_positive_am)
    n_l = n(l_res)
    n_r = n(r_res)
    cr = ufl.Circumradius(domain)
    h_l = 2 * cr(l_res)
    h_r = 2 * cr(r_res)

    # exchange current densities
    i0_n = kappa_elec * R * T / (Wa_n * faraday_const * characteristic_length)
    i0_p = kappa_elec * R * T / (Wa_p * faraday_const * characteristic_length)

    # concentration problem
    VC = fem.functionspace(submesh_positive_am, ("CG", 2))
    c, q = fem.Function(VC), ufl.TestFunction(VC)
    c0 = fem.Function(VC)
    cmax = 27000
    c0.interpolate(lambda x: x[0] - x[0] + 0.75*cmax)
    # c.interpolate(lambda x: x[0] - x[0] + 0.25 * cmax)
    # c.interpolate(c0)
    q_r = ufl.TestFunction(c.function_space)(r_res)
    q_l = ufl.TestFunction(c.function_space)(l_res)
    c_r = c(r_res)

    jump_u = surface_overpotential(kappa_pos_am, u_r, n_r, i0_p, kinetics_type=args.kinetics) + ocv_simple(c(r_res), cmax=cmax)

    F_0 = (
        -1/2 * mixed_term(kappa_elec * u_l + kappa_pos_am * u_r, v_l, n_l) * dInterface
        - 0.5 * mixed_term(0.5 * (kappa_elec + kappa_pos_am) * v_l, (u_r - u_l - jump_u), n_l) * dInterface
    )

    F_1 = (
        +1/2 * mixed_term(kappa_elec * u_l + kappa_pos_am * u_r, v_r, n_l) * dInterface
        - 0.5 * mixed_term(0.5 * (kappa_elec + kappa_pos_am) * v_r, (u_r - u_l - jump_u), n_l) * dInterface
    )
    F_0 += 2 * gamma / (h_l + h_r) * 0.5 * (kappa_elec + kappa_pos_am) * (u_r - u_l - jump_u) * v_l * dInterface
    F_1 += -2 * gamma / (h_l + h_r) * 0.5 * (kappa_elec + kappa_pos_am) * (u_r - u_l - jump_u) * v_r * dInterface

    F_0 += F_00
    F_1 += F_11

    F_2 = (c - c0)/dt * q * dx_r + D * inner(0.5 * ufl.grad(c + c0), ufl.grad(q)) * dx_r
    # F_2 += inner(i0_p/(R * T) * (u_r - u_l - ocv_simple(c(r_res), cmax=cmax)), q_r) * dInterface
    # F_2 += 1/faraday_const * inner(kappa_pos_am * grad(u_r), n_r) * q_r * dInterface
    # F_2 += -inner(D*grad(c(r_res)), n_r) * q_r * dInterface + i0_p/(R * T) * (u_r - u_l - ocv_simple(c(r_res), cmax=cmax)) * q_r * dInterface
    # F_2 += - gamma * h_r * inner(inner(D * grad(c(r_res)), n_r), inner(grad(q_r), n_r)) * dInterface
    # F_2 -= - gamma * h_r *  i0_p/(R * T) * (u_r - u_l - ocv_simple(c(r_res), cmax=cmax)) * inner(grad(q_r), n_r) * dInterface

    jac00 = ufl.derivative(F_0, u_0)
    jac01 = ufl.derivative(F_0, u_1)
    jac02 = ufl.derivative(F_0, c)

    jac10 = ufl.derivative(F_1, u_0)
    jac11 = ufl.derivative(F_1, u_1)
    jac12 = ufl.derivative(F_1, c)

    jac20 = ufl.derivative(F_2, u_0)
    jac21 = ufl.derivative(F_2, u_1)
    jac22 = ufl.derivative(F_2, c)

    J00 = fem.form(jac00, entity_maps=entity_maps)
    J01 = fem.form(jac01, entity_maps=entity_maps)
    J02 = fem.form(jac02, entity_maps=entity_maps)

    J10 = fem.form(jac10, entity_maps=entity_maps)
    J11 = fem.form(jac11, entity_maps=entity_maps)
    J12 = fem.form(jac12, entity_maps=entity_maps)

    J20 = fem.form(jac20, entity_maps=entity_maps)
    J21 = fem.form(jac21, entity_maps=entity_maps)
    J22 = fem.form(jac22, entity_maps=entity_maps)
    
    J = [[J00, J01, J02], [J10, J11, J12], [J20, J21, J22]]
    F = [
        fem.form(F_0, entity_maps=entity_maps),
        fem.form(F_1, entity_maps=entity_maps),
        fem.form(F_2, entity_maps=entity_maps),
    ]
    left_bc = fem.Function(u_0.function_space)
    left_bc.x.array[:] = 0
    submesh_electrolyte.topology.create_connectivity(
        submesh_electrolyte.topology.dim - 1, submesh_electrolyte.topology.dim
    )
    bc_left = fem.dirichletbc(
        left_bc, fem.locate_dofs_topological(u_0.function_space, fdim, ft_electrolyte.find(markers.left))
    )


    right_bc = fem.Function(u_1.function_space)
    right_bc.x.array[:] = args.voltage
    submesh_positive_am.topology.create_connectivity(
        submesh_positive_am.topology.dim - 1, submesh_positive_am.topology.dim
    )
    bc_right = fem.dirichletbc(
        right_bc, fem.locate_dofs_topological(u_1.function_space, fdim, ft_positive_am.find(markers.right))
    )
    bcs = [bc_left, bc_right]


    solver = solvers.NewtonSolver(
        F,
        J,
        [u_0, u_1, c],
        bcs=bcs,
        max_iterations=1000,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "superlu_dist",
        },
    )
    time = 0
    cvtx = io.VTXWriter(comm, concentration_file, [c], engine="BP5")
    for i in range(1):
        time += dt
        print(time)
        solver.solve(tol=1e-6, beta=0.5)
        c0.x.array[:] = c.x.array
        cvtx.write(time)
        I_left = comm.allreduce(fem.assemble_scalar(fem.form(inner(kappa_elec * grad(u_0), n) * ds(markers.left), entity_maps=entity_maps)), op=MPI.SUM)
        I_right = comm.allreduce(fem.assemble_scalar(fem.form(inner(kappa_pos_am * grad(u_1), n) * ds(markers.right), entity_maps=entity_maps)), op=MPI.SUM)
        I_interface = comm.allreduce(fem.assemble_scalar(fem.form(inner(faraday_const * D * grad(c(r_res)), n_r) * dInterface, entity_maps=entity_maps)), op=MPI.SUM)
        # I_interface2 = comm.allreduce(fem.assemble_scalar(fem.form(inner(faraday_const * D * grad(c), n2) * ds_r(markers.electrolyte_v_positive_am))), op=MPI.SUM)
        print(f"Current left: {I_left:.3e} [A]")
        print(f"Current interface: {I_interface:.3e} [A]")
        # print(f"Current interface2: {I_interface2:.3e} [A]")
        print(f"Current right: {I_right:.3e} [A]")
    cvtx.close()

    # interpolate
    V = fem.functionspace(domain, ("DG", 1))
    u = fem.Function(V)
    u.interpolate(u_0, cells1=submesh_electrolyte_to_mesh, cells0=np.arange(len(submesh_electrolyte_to_mesh)))
    u.interpolate(u_1, cells1=submesh_positive_am_to_mesh, cells0=np.arange(len(submesh_positive_am_to_mesh)))
    u.x.scatter_forward()

    with io.VTXWriter(comm, output_potential_file, [u], engine="BP5") as vtx:
        vtx.write(0)

    with io.VTXWriter(comm, elec_potential_file, [u_0], engine="BP5") as vtx:
        vtx.write(0)

    with io.VTXWriter(comm, positive_am_potential_file, [u_1], engine="BP5") as vtx:
        vtx.write(0)

    # with io.VTXWriter(comm, concentration_file, [c], engine="BP5") as vtx:
    #     vtx.write(0)
