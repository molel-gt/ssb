#!/usr/bin/env python3

import os
import timeit

import argparse
import dolfinx
import logging
import numpy as np
import ufl

from dolfinx import cpp, fem, io, log, mesh, nls
from mpi4py import MPI
from petsc4py import PETSc

import commons, configs, constants


markers = commons.SurfaceMarkers()
phases = commons.Phases()
# Some constants
D_am = 5e-15
D_se = 0
# assume electronic conductivity of SE is 0 and ionic conductivity of AM is 0
# conductivity
kappa_am = 0
kappa_se = 0.1
sigma_am = 1e3
sigma_se = 1e-4
source_am = 0.01
source_se = 1e-4
i0 = 1e-2  # exchange current density
eta_s = 0.005  # surface overpotential
phi2 = 0.225
F_c = 96485  # Faraday constant
R = 8.314
T = 298
alpha_a = 0.5
alpha_c = 0.5
# potentials
phi_ref = 0.25
phi_term = 3.7
i_superficial = 1e-4


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Laminate Cell')
    parser.add_argument('--data_dir', help='Directory with tria.xdmf and tetr.xmf mesh files. Output files potential.xdmf and current.xdmf will be saved here', required=True, type=str)
    parser.add_argument('--grid_size', help='Lx-Ly-Lz', required=True)
    args = parser.parse_args()
    data_dir = args.data_dir
    # voltage = args.voltage
    comm = MPI.COMM_WORLD
    rank = comm.rank
    start_time = timeit.default_timer()
    loglevel = configs.get_configs()['LOGGING']['level']
    Lx, Ly, Lz = map(lambda val: int(val), args.grid_size.split("-"))
    FORMAT = f'%(asctime)s: %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(f'{data_dir}')
    logger.setLevel(loglevel)
    line_mesh_path = os.path.join(data_dir, 'line.xdmf')
    tria_mesh_path = os.path.join(data_dir, 'tria.xdmf')
    output_current_path = os.path.join(data_dir, 'current.xdmf')
    output_potential_path = os.path.join(data_dir, 'potential.xdmf')

    left_cc_marker = markers.left_cc
    right_cc_marker = markers.right_cc
    insulated_marker = markers.insulated

    with io.XDMFFile(MPI.COMM_WORLD, tria_mesh_path, "r") as xdmf:
        domain = xdmf.read_mesh(cpp.mesh.GhostMode.shared_facet, name="Grid")
        ct = xdmf.read_meshtags(domain, name="Grid")

    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
    with io.XDMFFile(MPI.COMM_WORLD, line_mesh_path, "r") as xdmf:
        ft = xdmf.read_meshtags(domain, name="Grid")

    x = ufl.SpatialCoordinate(domain)
    n = ufl.FacetNormal(domain)
    Q = fem.FunctionSpace(domain, ("DG", 0))
    kappa = fem.Function(Q)
    sigma = fem.Function(Q)
    d_eff = fem.Function(Q)
    se_cells = ct.find(phases.electrolyte)
    am_cells = ct.find(phases.active_material)
    kappa.x.array[am_cells] = np.full_like(am_cells, kappa_am, dtype=PETSc.ScalarType)
    kappa.x.array[se_cells]  = np.full_like(se_cells, kappa_se, dtype=PETSc.ScalarType)
    sigma.x.array[am_cells] = np.full_like(am_cells, sigma_am, dtype=PETSc.ScalarType)
    sigma.x.array[se_cells]  = np.full_like(se_cells, sigma_se, dtype=PETSc.ScalarType)
    d_eff.x.array[am_cells] = np.full_like(am_cells, D_am, dtype=PETSc.ScalarType)
    d_eff.x.array[se_cells]  = np.full_like(se_cells, D_se, dtype=PETSc.ScalarType)

    # Source and Flux Terms
    f = fem.Constant(domain, PETSc.ScalarType(0.0))
    g = fem.Constant(domain, PETSc.ScalarType(0.0))
    g_left_cc = fem.Constant(domain, PETSc.ScalarType(i_superficial))

    V = fem.FunctionSpace(domain, ("CG", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    def i_linear(phi1, phi2=phi2):
        return i0 * F_c * (phi1 - phi2) / R / T
    
    def i_tafel(phi1, phi2=phi2):
        return i0  * ufl.exp(alpha_a * F_c * (phi1 - phi2) / R / T)

    def i_bv(phi1, phi2=phi2):
        return i0  * (ufl.exp(alpha_a * F_c * (phi1 - phi2) / R / T) - ufl.exp(-alpha_c * F_c * (phi1 - phi2) / R / T))

    def i_butler_volmer(phi1, phi2=phi2):
        return i0  * (ufl.exp(alpha_a * F_c * (phi1 - phi2) / R / T) - ufl.exp(-alpha_c * F_c * (phi1 - phi2) / R / T))

    # left_cc_curr = -i_butler_volmer() / kappa

    fdim = domain.topology.dim - 1
    vol_tag = mesh.meshtags(domain, domain.topology.dim, ct.indices, ct.values)
    facet_tag = mesh.meshtags(domain, fdim, ft.indices, ft.values)
    dx = ufl.Measure('dx', domain=domain, subdomain_data=vol_tag)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
    u_left_cc = fem.Function(V)
    with u_left_cc.vector.localForm() as u0_loc:
        u0_loc.set(phi_ref)
    
    u_right_cc = fem.Function(V)
    with u_right_cc.vector.localForm() as u0_loc:
        u0_loc.set(3.7)

    left_cc_facet = ft.find(markers.left_cc)
    left_cc_dofs = fem.locate_dofs_topological(V, 1, left_cc_facet)
    left_cc = fem.dirichletbc(u_left_cc, fem.locate_dofs_topological(V, 1, left_cc_facet))
    right_cc_facet = ft.find(markers.right_cc)
    right_cc_dofs = fem.locate_dofs_topological(V, 1, right_cc_facet)
    right_cc = fem.dirichletbc(u_right_cc, fem.locate_dofs_topological(V, 1, right_cc_facet))


    uh = fem.Function(V)
    bcs = [left_cc, right_cc]

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + sigma * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ds(markers.insulated)

    options = {
               "ksp_type": "gmres",
               "pc_type": "hypre",
               "ksp_rtol": 1.0e-12
               }

    model = dolfinx.fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options=options)

    logger.debug('Solving problem..')
    uh = model.solve()

    with io.XDMFFile(comm, output_potential_path, "w") as file:
        file.write_mesh(domain)
        file.write_function(uh)

    grad_u = ufl.grad(uh)
    area_left_cc = fem.assemble_scalar(fem.form(1 * ds(markers.left_cc)))
    area_right_cc = fem.assemble_scalar(fem.form(1 * ds(markers.right_cc)))
    i_left_cc = (1/area_left_cc) * fem.assemble_scalar(fem.form((kappa + sigma) * ufl.sqrt(ufl.inner(grad_u, grad_u)) * ds(markers.left_cc)))
    i_right_cc = (1/area_right_cc) * fem.assemble_scalar(fem.form(sigma * ufl.sqrt(ufl.inner(grad_u, grad_u)) * ds(markers.right_cc)))

    W = fem.FunctionSpace(domain, ("Lagrange", 1))
    current_expr = fem.Expression((kappa + sigma) * ufl.sqrt(ufl.inner(grad_u, grad_u)), W.element.interpolation_points())
    current_h = fem.Function(W)
    current_h.interpolate(current_expr)

    with io.XDMFFile(comm, output_current_path, "w") as file:
        file.write_mesh(domain)
        file.write_function(current_h)

    print("Current density @ left cc                       : {:.4e}".format(i_left_cc))
    # print("Current density @ right cc                      : {:.4e}".format(i_right_cc))
