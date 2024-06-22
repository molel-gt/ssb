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

from basix.ufl import element
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

import commons, utils

warnings.simplefilter('ignore')

dtype = PETSc.ScalarType
kappa_elec = 0.1
faraday_const = 96485
R = 8.3145
T = 298

class NewtonSolver:
    max_iterations: int
    bcs: list[fem.DirichletBC]
    A: PETSc.Mat
    b: PETSc.Vec
    J: fem.Form
    b: fem.Form
    dx: PETSc.Vec

    def __init__(
        self,
        F: list[fem.form],
        J: list[list[fem.form]],
        w: list[fem.Function],
        bcs: list[fem.DirichletBC] | None = None,
        max_iterations: int = 5,
        petsc_options: dict[str, str | float | int | None] = None,
        problem_prefix="newton",
    ):
        self.max_iterations = max_iterations
        self.bcs = [] if bcs is None else bcs
        self.b = fem.petsc.create_vector_block(F)
        self.F = F
        self.J = J
        self.A = fem.petsc.create_matrix_block(J)
        self.dx = self.A.createVecLeft()
        self.w = w
        self.x = fem.petsc.create_vector_block(F)

        # Set PETSc options
        opts = PETSc.Options()
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v

        # Define KSP solver
        self._solver = PETSc.KSP().create(self.b.getComm().tompi4py())
        self._solver.setOperators(self.A)
        self._solver.setFromOptions()

        # Set matrix and vector PETSc options
        self.A.setFromOptions()
        self.b.setFromOptions()

    def solve(self, tol=1e-6, beta=1.0):
        i = 0

        while i < self.max_iterations:
            dolfinx.cpp.la.petsc.scatter_local_vectors(
                self.x,
                [si.x.petsc_vec.array_r for si in self.w],
                [
                    (
                        si.function_space.dofmap.index_map,
                        si.function_space.dofmap.index_map_bs,
                    )
                    for si in self.w
                ],
            )
            self.x.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

            # Assemble F(u_{i-1}) - J(u_D - u_{i-1}) and set du|_bc= u_D - u_{i-1}
            with self.b.localForm() as b_local:
                b_local.set(0.0)
            fem.petsc.assemble_vector_block(
                self.b, self.F, self.J, bcs=self.bcs, x0=self.x, scale=-1.0
            )
            self.b.ghostUpdate(
                PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD
            )


            # Assemble Jacobian
            self.A.zeroEntries()
            fem.petsc.assemble_matrix_block(self.A, self.J, bcs=self.bcs)
            self.A.assemble()

            self._solver.solve(self.b, self.dx)
            # self._solver.view()
            assert (
                self._solver.getConvergedReason() > 0
            ), "Linear solver did not converge"
            offset_start = 0
            for s in self.w:
                num_sub_dofs = (
                    s.function_space.dofmap.index_map.size_local
                    * s.function_space.dofmap.index_map_bs
                )
                s.x.petsc_vec.array_w[:num_sub_dofs] -= (
                    beta * self.dx.array_r[offset_start : offset_start + num_sub_dofs]
                )
                s.x.petsc_vec.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
                )
                offset_start += num_sub_dofs
            # Compute norm of update

            correction_norm = self.dx.norm(0)
            print(f"Iteration {i}: Correction norm {correction_norm}")
            if correction_norm < tol:
                break
            i += 1

    def __del__(self):
        self.A.destroy()
        self.b.destroy()
        self.dx.destroy()
        self._solver.destroy()
        self.x.destroy()


def arctanh(y):
    return 0.5 * ufl.ln((1 + y) / (1 - y))


def ocv(c, cmax=30000):
    xi = 2 * (c - 0.5 * cmax) / cmax
    return 3.25 - 0.5 * arctanh(xi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Current Collector.')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="lithium_metal_3d_cc_2d")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1.0, type=float)
    parser.add_argument("--Wa_n", help="Wagna number for negative electrode: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e-3, type=float)
    parser.add_argument("--Wa_p", help="Wagna number for positive electrode: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=1e3, type=float)
    parser.add_argument("--kr", help="ratio of ionic to electronic conductivity", nargs='?', const=1, default=1, type=float)
    parser.add_argument("--gamma", help="interior penalty parameter", nargs='?', const=1, default=15, type=float)
    parser.add_argument("--atol", help="solver absolute tolerance", nargs='?', const=1, default=1e-12, type=float)
    parser.add_argument("--rtol", help="solver relative tolerance", nargs='?', const=1, default=1e-9, type=float)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='MICRON_TO_METER', type=str)

    args = parser.parse_args()
    start_time = timeit.default_timer()
    voltage = args.voltage
    Wa_n = args.Wa_n
    Wa_p = args.Wa_p
    comm = MPI.COMM_WORLD
    encoding = io.XDMFFile.Encoding.HDF5
    micron = 1e-6
    LX, LY, LZ = [float(vv) * micron for vv in args.dimensions.split("-")]
    workdir = os.path.join(args.mesh_folder, str(Wa_n) + "-" + str(Wa_p) + "-" + str(args.kr), str(args.gamma))
    utils.make_dir_if_missing(workdir)
    output_meshfile = os.path.join(args.mesh_folder, 'mesh.msh')
    lines_h5file = os.path.join(args.mesh_folder, 'lines.h5')
    potential_resultsfile = os.path.join(workdir, "potential.bp")
    concentration_resultsfile = os.path.join(workdir, "concentration.bp")
    current_dist_file = os.path.join(workdir, f"current-y-positions-{str(args.Wa_p)}-{str(args.kr)}.png")
    reaction_dist_file = os.path.join(workdir, f"reaction-dist-{str(args.Wa_p)}-{str(args.kr)}.png")
    current_resultsfile = os.path.join(workdir, "current.bp")
    simulation_metafile = os.path.join(workdir, "simulation.json")

    markers = commons.Markers()
    kappa_pos_am = kappa_elec / args.kr

    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = gmshio.read_from_msh(output_meshfile, comm, partitioner=partitioner)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)

    ft_imap = domain.topology.index_map(fdim)
    num_facets = ft_imap.size_local + ft_imap.num_ghosts
    indices = np.arange(0, num_facets)
    values = np.zeros(indices.shape, dtype=np.intc)  # all facets are tagged with zero

    values[ft.indices] = ft.values
    ft = mesh.meshtags(domain, fdim, indices, values)

    # create submesh
    submesh, entity_map, vertex_map, geom_map = mesh.create_submesh(
        domain, tdim, ct.find(markers.positive_am)
    )
    # transfer tags from parent to submesh
    tdim = domain.topology.dim
    fdim = tdim - 1
    c_to_f = domain.topology.connectivity(tdim, fdim)
    f_map = domain.topology.index_map(fdim)
    all_facets = f_map.size_local + f_map.num_ghosts
    all_values = np.zeros(all_facets, dtype=np.int32)
    all_values[ft.indices] = ft.values

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
    entity_maps = {submesh: mesh_to_submesh, domain: ct.indices}

    # integration measures
    dx = ufl.Measure("dx", domain=domain, subdomain_data=ct)
    dx_r = ufl.Measure("dx", domain=submesh)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    ds_r = ufl.Measure("ds", domain=submesh, subdomain_data=submesh_ft)

    f_to_c = domain.topology.connectivity(fdim, tdim)
    c_to_f = domain.topology.connectivity(tdim, fdim)
    charge_xfer_facets = ft.find(markers.electrolyte_v_positive_am)

    other_internal_facets = ft.find(0)
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

    other_internal_facet_domains = []
    for f in other_internal_facets:
        if f >= ft_imap.size_local or len(f_to_c.links(f)) != 2:
            continue
        c_0, c_1 = f_to_c.links(f)[0], f_to_c.links(f)[1]
        subdomain_0, subdomain_1 = ct.values[[c_0, c_1]]
        local_f_0 = np.where(c_to_f.links(c_0) == f)[0][0]
        local_f_1 = np.where(c_to_f.links(c_1) == f)[0][0]
        other_internal_facet_domains.append(c_0)
        other_internal_facet_domains.append(local_f_0)
        other_internal_facet_domains.append(c_1)
        other_internal_facet_domains.append(local_f_1)
    int_facet_domains = [(markers.electrolyte_v_positive_am, int_facet_domain)]#, (0, other_internal_facet_domains)]

    dS = ufl.Measure("dS", domain=domain, subdomain_data=int_facet_domains)

    # concentrations for nmc622
    nmc_622_density = 2.597e3
    area_electrolyte = fem.assemble_scalar(fem.form(1.0 * dx(markers.electrolyte)))
    area_positive_am = fem.assemble_scalar(fem.form(1.0 * dx(markers.positive_am)))
    print(f"Electrolyte Area: {area_electrolyte:.2e}")
    print(f"Positive AM Area: {area_positive_am:.2e}")
    # assume 1 meter
    positive_capacity = utils.nmc_capacity(nmc_622_density, area_positive_am * 1)
    print(f"Positive Capacity: {positive_capacity * 1000:.3f} [mA.h]")
    c_li = utils.lithium_concentration_nmc(nmc_622_density)
    print(f"Lithium concentration in NMC622: {c_li} [mol/m3]")


    # ### Function Spaces

    V = fem.functionspace(domain, ("DG", 1))
    V_submesh = fem.functionspace(submesh, ("CG", 1))
    W = fem.functionspace(domain, ("DG", 1, (3,)))
    Q = fem.functionspace(domain, ("DG", 0))
    u = fem.Function(V, name='potential')
    c = fem.Function(V_submesh, name='concentration')
    c0 = fem.Function(V_submesh, name='concentration')
    c0.interpolate(lambda x: 0.5 * c_li + x[0] - x[0])
    c.interpolate(c0)
    q = ufl.TestFunction(V_submesh)
    # u_cg = fem.Function(V_CG, name='potential', dtype=np.float64)
    v = ufl.TestFunction(V)
    n = ufl.FacetNormal(domain)
    nc = ufl.FacetNormal(submesh)
    x = ufl.SpatialCoordinate(domain)

    h = ufl.CellDiameter(domain)
    h_avg = avg(h)


    # constants

    f = fem.Constant(domain, dtype(0))
    fc = fem.Constant(submesh, dtype(0))
    g = fem.Constant(domain, dtype(0))
    gc = fem.Constant(submesh, dtype(0))
    u_left = fem.Function(V)
    with u_left.vector.localForm() as u0_loc:
        u0_loc.set(0)
    u_right = fem.Function(V)
    with u_right.vector.localForm() as u1_loc:
        u1_loc.set(voltage)

    # #### $\kappa$ varying in each domain

    kappa = fem.Function(Q)
    cells_electrolyte = ct.find(markers.electrolyte)
    cells_pos_am = ct.find(markers.positive_am)
    kappa.x.array[cells_electrolyte] = np.full_like(cells_electrolyte, kappa_elec, dtype=dtype)
    kappa.x.array[cells_pos_am] = np.full_like(cells_pos_am, kappa_pos_am, dtype=dtype)
    D = 1e-15

    # ### variational formulation

    alpha = 100  # args.gamma
    gamma = 100  # args.gamma

    i0_n = kappa_elec * R * T / (args.Wa_n * faraday_const * LX)
    i0_p = kappa_elec * R * T / (args.Wa_p * faraday_const * LX)

    # potential problem
    i_loc = -inner((kappa * grad(u))('+'), n("+"))
    u_jump = 2 * ufl.ln(0.5 * i_loc/i0_p + ufl.sqrt((0.5 * i_loc/i0_p)**2 + 1)) * (R * T / faraday_const)

    u_ocv = ocv(c("+"))
    F0 = kappa * inner(grad(u), grad(v)) * dx - f * v * dx - kappa * inner(grad(u), n) * v * ds

    # Add DG/IP terms
    F0 += - avg(kappa) * inner(jump(u, n), avg(grad(v))) * dS#(0)
    F0 += - inner(avg(kappa * grad(u)), jump(v, n)) * dS#(0)
    F0 += alpha / h_avg * avg(kappa) * inner(jump(v, n), jump(u, n)) * dS#(0)

    # Internal boundary
    F0 += + avg(kappa) * dot(avg(grad(v)), (u_jump + u_ocv) * n('+')) * dS(markers.electrolyte_v_positive_am)
    F0 += -alpha / h_avg * avg(kappa) * dot(jump(v, n), (u_jump + u_ocv) * n('+')) * dS(markers.electrolyte_v_positive_am)

    # # Symmetry
    F0 += - avg(kappa) * inner(jump(u, n), avg(grad(v))) * dS(markers.electrolyte_v_positive_am)

    # # Coercivity
    F0 += alpha / h_avg * avg(kappa) * inner(jump(u, n), jump(v, n)) * dS(markers.electrolyte_v_positive_am)

    # Nitsche Dirichlet BC terms on left and right boundaries
    F0 += - kappa * (u - u_left) * inner(n, grad(v)) * ds(markers.left)
    F0 += -gamma / h * (u - u_left) * v * ds(markers.left)
    F0 += - kappa * (u - u_right) * inner(n, grad(v)) * ds(markers.right) 
    F0 += -gamma / h * (u - u_right) * v * ds(markers.right)

    # Nitsche Neumann BC terms on insulated boundary
    F0 += -g * v * ds(markers.insulated_electrolyte) + gamma * h * g * inner(grad(v), n) * ds(markers.insulated_electrolyte)
    F0 += - gamma * h * inner(inner(grad(u), n), inner(grad(v), n)) * ds(markers.insulated_electrolyte)
    F0 += -g * v * ds(markers.insulated_positive_am) + gamma * h * g * inner(grad(v), n) * ds(markers.insulated_positive_am)
    F0 += - gamma * h * inner(inner(grad(u), n), inner(grad(v), n)) * ds(markers.insulated_positive_am)

    # kinetics boundary - neumann
    # F += - gamma * h * inner(inner(kappa * grad(u), n), inner(grad(v), n)) * ds(markers.left)
    # F -= - gamma * h * 2 * i0_n * ufl.sinh(0.5 * faraday_const / R / T * (V_left - u - 0)) * inner(grad(v), n) * ds(markers.left)

    # concentration problem

    dt = 1e-3
    F1 = inner(c - c0, q) * dx_r + dt * inner(D * grad(c), grad(q)) * dx_r
    F1 -= dt * (inner(fc, q) * dx_r + inner(gc, q) * (ds_r(markers.insulated_positive_am) + ds_r(markers.right)) - inner(1/faraday_const * inner(-kappa * grad(u), n), q) * ds_r(markers.electrolyte_v_positive_am))


    # solve tertiary current distribution

    TIME = 500 * dt
    t = 0
    c_vtx = VTXWriter(comm, concentration_resultsfile, [c], engine="BP5")
    u_vtx = VTXWriter(comm, potential_resultsfile, [u], engine="BP5")
    c_vtx.write(0.0)

    while t < TIME:
        print(f"Time: {t:.3f}")
        t += dt
        jac00 = ufl.derivative(F0, u)
        jac01 = ufl.derivative(F0, c)
        jac10 = ufl.derivative(F1, u)
        jac11 = ufl.derivative(F1, c)
        
        J00 = fem.form(jac00, entity_maps=entity_maps)
        J01 = fem.form(jac01, entity_maps=entity_maps)
        J10 = fem.form(jac10, entity_maps=entity_maps)
        J11 = fem.form(jac11, entity_maps=entity_maps)
        
        J = [[J00, J01], [J10, J11]]
        F = [
            fem.form(F0, entity_maps=entity_maps),
            fem.form(F1, entity_maps=entity_maps),
            ]
        solver = NewtonSolver(
            F,
            J,
            [u, c],
            bcs=[],
            max_iterations=10,
            petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "superlu_dist",
            },
            )
        solver.solve(1e-6, beta=1)
        c0.x.array[:] = c.x.array[:]
        c_vtx.write(t)
        u_vtx.write(t)
    c_vtx.close()
    u_vtx.close()

    # summary statistics
    current_h = fem.Function(W)
    current_expr = fem.Expression(-kappa * grad(u), W.element.interpolation_points())
    current_h.interpolate(current_expr)
    with VTXWriter(comm, current_resultsfile, [current_h], engine="BP5") as vtx:
        vtx.write(0.0)

    I_neg_charge_xfer = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(current_h, n) * ds(markers.left))), op=MPI.SUM)
    I_pos_am = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(current_h("+"), n("+")) * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)
    I_right = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(current_h, n) * ds(markers.right))), op=MPI.SUM)
    I_insulated_elec = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(inner(current_h, n)) * ds(markers.insulated_electrolyte))), op=MPI.SUM)
    I_insulated_pos_am = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(inner(current_h, n)) * ds(markers.insulated_positive_am))), op=MPI.SUM)
    I_insulated = I_insulated_elec + I_insulated_pos_am
    area_left = domain.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * ds(markers.left))), op=MPI.SUM)
    area_neg_charge_xfer = domain.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * ds(markers.left))), op=MPI.SUM)
    area_pos_charge_xfer = domain.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)
    area_right = domain.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * ds(markers.right))), op=MPI.SUM)
    i_sup_left = np.abs(I_neg_charge_xfer / area_neg_charge_xfer)
    i_sup = np.abs(I_right / area_right)
    i_pos_am = I_pos_am / area_pos_charge_xfer
    std_dev_i_pos_am = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((np.abs(inner(current_h("+"), n("+"))) - np.abs(i_pos_am)) ** 2 * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)/area_pos_charge_xfer)
    std_dev_i_pos_am_norm = np.abs(std_dev_i_pos_am / i_pos_am)
    eta_p = domain.comm.allreduce(fem.assemble_scalar(fem.form(2 * R * T / (faraday_const) * utils.arcsinh(0.5 * np.abs(-inner(kappa * grad(u), n)("+"))) * dS(markers.electrolyte_v_positive_am))), op=MPI.SUM)
    u_avg_right = domain.comm.allreduce(fem.assemble_scalar(fem.form(u * ds(markers.right))) / area_right, op=MPI.SUM)
    u_avg_left = domain.comm.allreduce(fem.assemble_scalar(fem.form(u * ds(markers.left))) / area_left, op=MPI.SUM)
    u_stdev_right = domain.comm.allreduce(np.sqrt(fem.assemble_scalar(fem.form((u - u_avg_right) ** 2 * ds(markers.right))) / area_right), op=MPI.SUM)
    u_stdev_left = domain.comm.allreduce(np.sqrt(fem.assemble_scalar(fem.form((u - u_avg_left) ** 2 * ds(markers.left))) / area_left), op=MPI.SUM)
    eta_n = u_avg_left
    simulation_metadata = {
        "Negative Wagner Number": f"{Wa_n:.1e}",
        "Positive Wagner Number": f"{Wa_p:.1e}",
        "Negative Overpotential [V]": f"{eta_n:.2e}",
        "Positive Overpotential [V]": f"{eta_p:.2e}",
        "Voltage": voltage,
        "dimensions": args.dimensions,
        "interior penalty (gamma)": args.gamma,
        "interior penalty kr-modified (gamma)": gamma,
        "ionic to electronic conductivity ratio (kr)": args.kr,
        "average potential left [V]": f"{u_avg_left:.2e}",
        "stdev potential left [V]": f"{u_stdev_left:.2e}",
        "average potential right [V]": f"{u_avg_right:.2e}",
        "stdev potential right [V]": f"{u_stdev_right:.2e}",
        "Superficial current density [A/m2]": f"{np.abs(i_sup):.2e} [A/m2]",
        "Current at negative am - electrolyte boundary": f"{np.abs(I_neg_charge_xfer):.2e} A",
        "Current at electrolyte - positive am boundary": f"{np.abs(I_pos_am):.2e} A",
        "Current at right boundary": f"{np.abs(I_right):.2e} A",
        "Current at insulated boundary": f"{I_insulated:.2e} A",
        "stdev i positive charge transfer": f"{std_dev_i_pos_am:.2e} A/m2",
        "stdev i positive charge transfer (normalized)": f"{std_dev_i_pos_am_norm:.2e}",
        "solver atol": args.atol,
        "solver rtol": args.rtol,

    }
    if comm.rank == 0:
        utils.print_dict(simulation_metadata, padding=50)
        with open(simulation_metafile, "w", encoding='utf-8') as f:
            json.dump(simulation_metadata, f, ensure_ascii=False, indent=4)
        print(f"Saved results files in {workdir}")
        print(f"Time elapsed: {int(timeit.default_timer() - start_time):3.5f}s")
