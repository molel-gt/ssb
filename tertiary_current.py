# SPDX-License-Identifier: MIT
import argparse
import json
import os
import resource
import time
import timeit

import datetime
import dolfinx
import dolfinx.fem.petsc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matspy
import numpy as np
import scipy
import scipy.special as sp
import ufl

from dolfinx import cpp, fem, io, mesh
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
from ufl import dot, grad, inner

import commons, constants, mesh_utils, solvers, utils 


R = 8.314
T = 298
faraday_const = 96485
kappa_pos_am = 0.1
kinetics = ('linear', 'tafel', 'butler_volmer')
micron = 1e-6
V_UCO = 4.25  # upper cutoff voltage
c_max = 35000


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


def surface_overpotential(kappa, u, n, i0, kinetics_type='linear', ref={"L": 1, "phi": 1, "t": 1, "c": 1}):
    i_loc = -inner((kappa * grad(u)), n) * ref["phi"]/ref["L"]
    if kinetics_type == "butler_volmer":
        return 2 * ufl.ln(0.5 * i_loc/i0 + ufl.sqrt((0.5 * i_loc/i0)**2 + 1)) * (R * T / (faraday_const * ref["phi"]))
    elif kinetics_type == "linear":
        return R * T * i_loc / (i0 * faraday_const * ref["phi"])
    elif kinetics_type == "tafel":
        return ufl.sign(i_loc) * R * T / (0.5 * faraday_const * ref["phi"]) * ufl.ln(np.abs(i_loc)/i0)


def arctanh(y):
    return 0.5 * ufl.ln((1 + y) / (1 - y))


def ocv(c, cmax=35000):
    xi = 2 * (c - 0.5 * cmax) / cmax
    return 3.25 - 0.5 * arctanh(xi)
    # return 3.25 - 0.25 * ufl.ln((1 + 2 * (c - 0.5 * cmax) / cmax) / (1 - 2 * (c - 0.5 * cmax) / cmax))


def ocv_simple(c, cmax=35000):
    # return 3.25 + (1.125-c/cmax)**0.5 - (c/cmax)**0.5 + ufl.sinh(1-c/cmax)
    return 2.25*(1/ufl.cosh(1 - c/cmax) + ufl.sinh(1 - c/cmax))


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
    dt_ = 1e-6
    D = 1e-15
    TIME = 1 * dt_

    markers = commons.Markers()
    comm = MPI.COMM_WORLD

    dimensions = utils.extract_dimensions_from_meshfolder(args.mesh_folder)
    LX, LY, LZ = [float(vv) * micron for vv in dimensions.split("-")]

    L_ref = LZ
    A0 = LX * LY * 1e4  # [cm^2]
    if np.isclose(LZ, 0):
        A0 = LX * 1e4  # [cm^2]
        L_ref = LX

    # reference values
    t_ref = L_ref ** 2 / D
    phi_ref = V_UCO
    c_ref = c_max
    ref = {"t": t_ref, "phi": phi_ref, "c": c_ref, "L": L_ref}

    output_meshfile = os.path.join(args.mesh_folder, "mesh.msh")
    results_dir = os.path.join(args.mesh_folder, args.kinetics, str(Wa_n) + "-" + str(Wa_p) + "-" + str(args.kr), str(args.gamma))
    utils.make_dir_if_missing(results_dir)
    output_potential_file = os.path.join(results_dir, "potential.bp")
    elec_potential_file = os.path.join(results_dir, "electrolyte_potential.bp")
    positive_am_potential_file = os.path.join(results_dir, "positive_am_potential.bp")
    current_file = os.path.join(results_dir, "current.bp")
    concentration_file = os.path.join(results_dir, "concentration.bp")
    simulation_metafile = os.path.join(results_dir, "simulation.json")
    resource_usage = os.path.join(results_dir, f"resources-{comm.Get_rank()}.log")
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    with open(resource_usage, 'a') as f:
        # Dump timestamp, PID and amount of RAM.
        f.write('{} {} {}\n'.format(datetime.datetime.now(), os.getpid(), mem))

    # load mesh
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = io.gmshio.read_from_msh(output_meshfile, comm, partitioner=partitioner)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)
    domain.topology.create_connectivity(tdim, tdim)
    domain.topology.create_connectivity(fdim, fdim)
    with open(resource_usage, 'a') as f:
        # Dump timestamp, PID and amount of RAM.
        f.write('{} {} {}\n'.format(datetime.datetime.now(), os.getpid(), mem))

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
    with open(resource_usage, 'a') as f:
        # Dump timestamp, PID and amount of RAM.
        f.write('{} {} {}\n'.format(datetime.datetime.now(), os.getpid(), mem))


    u_0, F_00, m_to_elec = define_interior_eq(domain, 1, submesh_electrolyte, submesh_electrolyte_to_mesh, 0.0, kappa_elec)
    u_1, F_11, m_to_pos_am = define_interior_eq(domain, 1, submesh_positive_am, submesh_positive_am_to_mesh, 0.0, kappa_pos_am)
    u_0.name = "u_b"
    u_1.name = "u_t"

    # initial guess
    u_0.interpolate(lambda x: x[0]-x[0] + 0)
    u_1.interpolate(lambda x: x[0]-x[0] + 1)

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
    dx_r = ufl.Measure('dx', domain=domain, subdomain_data=ct, subdomain_id=markers.positive_am)
    ds = ufl.Measure('ds', domain=domain, subdomain_data=ft)
    ds_r = ufl.Measure('ds', domain=submesh_positive_am, subdomain_data=ft_positive_am)
    l_res = "-"
    r_res = "+"
    V0 = u_0.function_space
    V1 = u_1.function_space

    v_l = ufl.TestFunction(V0)(l_res)
    v_r = ufl.TestFunction(V1)(r_res)
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
    i0_n = kappa_elec * R * T / (Wa_n * faraday_const * L_ref)
    i0_p = kappa_elec * R * T / (Wa_p * faraday_const * L_ref)

    # concentration problem
    dt = fem.Constant(submesh_positive_am, dt_)
    VC = fem.functionspace(submesh_positive_am, ("CG", 4))

    c, q = fem.Function(VC), ufl.TestFunction(VC)
    c0 = fem.Function(VC)

    c0.interpolate(lambda x: x[0] - x[0] + 0.75)
    c.interpolate(c0)

    q_r = ufl.TestFunction(c.function_space)(r_res)
    q_l = ufl.TestFunction(c.function_space)(l_res)
    c_r = c(r_res)

    jump_u = surface_overpotential(kappa_pos_am, u_r, n_r, i0_p, kinetics_type=args.kinetics, ref=ref) + ocv_simple(c(r_res), cmax=1)/phi_ref

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

    F_2 = (c - c0)/dt * q * dx_r + inner(ufl.grad(c), ufl.grad(q)) * dx_r
    F_2 += -inner(kappa_pos_am * phi_ref/(D * faraday_const * c_ref) * grad(u_r), n_r) * q_r * dInterface

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

    ###################### sparsity structure ##################################
    if args.plot:
        J_diag = fem.petsc.assemble_matrix_block([[J00, None, None], [None, J11, None], [None, None, J22]])
        J_off_diag = fem.petsc.assemble_matrix_block([[None, J01, J02], [J10, None, J12], [J20, J21, None]])
        J_diag.assemble()
        J_off_diag.assemble()
        ai0, aj0, av0 = J_diag.getValuesCSR()
        ai1, aj1, av1 = J_off_diag.getValuesCSR()
        Asp0 = scipy.sparse.csr_matrix((av0, aj0, ai0))
        Asp1 = scipy.sparse.csr_matrix((av1, aj1, ai1))
        mpl.rcParams['savefig.pad_inches'] = 0
        mpl.rcParams['figure.figsize'] = (5, 4.5)
        matspy.params.title = False
        matspy.params.indices = False
        matspy.shading = False
        fig0, ax0 = matspy.spy_to_mpl(Asp0)
        fig1, ax1 = matspy.spy_to_mpl(Asp1)
        ax0.set_box_aspect(1);
        ax0.axis('off');
        fig0.frameon = False
        fig0.savefig(os.path.join(results_dir, "jacobian-diag-sparsity.eps"), bbox_inches='tight', transparent=True)
        ax1.set_box_aspect(1);
        ax1.axis('off');
        fig1.frameon = False
        fig1.savefig(os.path.join(results_dir, "jacobian-off-diag-sparsity.eps"), bbox_inches='tight', transparent=True)
    ############################################################################
    F = [
        fem.form(F_0, entity_maps=entity_maps),
        fem.form(F_1, entity_maps=entity_maps),
        fem.form(F_2, entity_maps=entity_maps),
    ]
    left_bc = fem.Function(u_0.function_space)
    left_bc.x.array[:] = 0/phi_ref
    submesh_electrolyte.topology.create_connectivity(
        submesh_electrolyte.topology.dim - 1, submesh_electrolyte.topology.dim
    )
    bc_left = fem.dirichletbc(
        left_bc, fem.locate_dofs_topological(u_0.function_space, fdim, ft_electrolyte.find(markers.left))
    )


    right_bc = fem.Function(u_1.function_space)
    right_bc.x.array[:] = args.voltage/phi_ref
    submesh_positive_am.topology.create_connectivity(
        submesh_positive_am.topology.dim - 1, submesh_positive_am.topology.dim
    )
    bc_right = fem.dirichletbc(
        right_bc, fem.locate_dofs_topological(u_1.function_space, fdim, ft_positive_am.find(markers.right))
    )
    bcs = [bc_left, bc_right]

    with open(resource_usage, 'a') as f:
        # Dump timestamp, PID and amount of RAM.
        f.write('{} {} {}\n'.format(datetime.datetime.now(), os.getpid(), mem))
    V0_map = V0.dofmap.index_map
    V1_map = V1.dofmap.index_map
    VC_map = VC.dofmap.index_map
    # offset_u1 = V0_map.size_local*V0.dofmap.index_map_bs
    offset_c = V0_map.size_local*V0.dofmap.index_map_bs + V1_map.size_local*V1.dofmap.index_map_bs
    n_dofs = V0_map.size_global*V0.dofmap.index_map_bs + V1_map.size_global*V1.dofmap.index_map_bs + VC_map.size_global*VC.dofmap.index_map_bs
    t = 0
    cvtx = io.VTXWriter(comm, concentration_file, [c], engine="BP5")
    while t < TIME:
        t += dt.value
        PETSc.Sys.Print(f"Time: {t:.1e}\n")
        Jmat = fem.petsc.create_matrix_block(J)
        Fvec = fem.petsc.create_vector_block(F)
        P_0 = [[J00, None, None], [None, J11, None], [None, None, J22]]
        P = fem.petsc.assemble_matrix_block(P_0, bcs=bcs)
        P.assemble()
        snes = PETSc.SNES().create(comm)
        # snes.getKSP().setOperators(Jmat, None)
        snes.setTolerances(rtol=1.0e-7, max_it=100)
        # snes.setMonitor(lambda _, it, residual: print(it, residual))
        snes.setErrorIfNotConverged(True)
        snes.getKSP().setErrorIfNotConverged(True)
        snes.setType('newtonls')
        snes.getKSP().setType("preonly")
        snes.getKSP().getPC().setType("lu")
        snes.getKSP().getPC().setFactorSolverType("mumps")
        opts = PETSc.Options()
        opts['snes_linesearch_type'] = 'bt'
        opts['snes_monitor'] = None
        opts['snes_linesearch_monitor'] = None
        snes.setFromOptions()
        snes.getKSP().setFromOptions()

        with open(resource_usage, 'a') as f:
            # Dump timestamp, PID and amount of RAM.
            f.write('{} {} {}\n'.format(datetime.datetime.now(), os.getpid(), mem))

        problem = solvers.NonlinearPDE_SNESProblem(F, J, [u_0, u_1, c], bcs)
        snes.setFunction(problem.F_block, Fvec)
        snes.setJacobian(problem.J_block, J=Jmat, P=None)
        snes.getKSP().view()
        snes.view()
        x = fem.petsc.create_vector_block(F)
        cpp.la.petsc.scatter_local_vectors(
            x,
            [u_0.x.petsc_vec.array_r, u_1.x.petsc_vec.array_r, c.x.petsc_vec.array_r],
            [
                (u_0.function_space.dofmap.index_map, u_0.function_space.dofmap.index_map_bs),
                (u_1.function_space.dofmap.index_map, u_1.function_space.dofmap.index_map_bs),
                (c.function_space.dofmap.index_map, c.function_space.dofmap.index_map_bs),
            ],
        )
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        t0 = time.time()
        snes.solve(None, x)
        t1 = time.time()
        # print(snes.getConvergedReason(), snes.getKSP().getConvergedReason())
        assert snes.getKSP().getConvergedReason() > 0
        assert snes.getConvergedReason() > 0
        xnorm = x.norm()
        snes.destroy()
        Jmat.destroy()
        Fvec.destroy()
        x.destroy()
        c0.x.array[:] = c.x.array
        cvtx.write(t)
        I_left = comm.allreduce(fem.assemble_scalar(fem.form(inner(kappa_elec * (phi_ref) * L_ref ** (tdim-2) * grad(u_0), n) * ds(markers.left), entity_maps=entity_maps)), op=MPI.SUM)
        I_right = comm.allreduce(fem.assemble_scalar(fem.form(inner(kappa_pos_am * (phi_ref) * L_ref ** (tdim-2) * grad(u_1), n) * ds(markers.right), entity_maps=entity_maps)), op=MPI.SUM)
        I_interface = comm.allreduce(fem.assemble_scalar(fem.form(inner(faraday_const * D * (c_ref) * L_ref ** (tdim-2) * grad(c(r_res)), n_r) * dInterface, entity_maps=entity_maps)), op=MPI.SUM)
        PETSc.Sys.Print(
                f"Current left: {I_left:.3e} [A]\n"
                f"Current interface: {I_interface:.3e} [A]\n"
                f"Current right: {I_right:.3e} [A]\n"
                )
    cvtx.close()
    with open(resource_usage, 'a') as f:
    # Dump timestamp, PID and amount of RAM.
        f.write('{} {} {}\n'.format(datetime.datetime.now(), os.getpid(), mem))

    time_elapsed = timeit.default_timer() - start_time

    resistance = args.voltage / (np.abs(I_left) * A0)
    metadata = {
        "I left [A]": I_left,
        "I interface [A]": I_interface,
        "I right [A]": I_right,
        "resistance [ohm.cm2]": resistance,
        "time elapsed [s]": time_elapsed,
        "solve time [s]": t1 - t0,
        "L ref [m]": ref["L"],
        "c ref [mol/m3]": ref["c"],
        "phi ref [V]": ref["phi"],
        "t ref [s]": ref["t"],
        "Positive Wa": args.Wa_p,
        "Kr": args.kr,
        "kinetics": args.kinetics,
        "dofs": n_dofs,
    }
    if comm.rank == 0:
        utils.print_dict(metadata, padding=50)
        with open(simulation_metafile, "w", encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        PETSc.Sys.Print(f"Saved results files in {results_dir}")
        PETSc.Sys.Print(f"Time elapsed: {time_elapsed:3.5f}s")

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
