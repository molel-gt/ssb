#!/usr/bin/env python3

import os
import pickle
import timeit

import argparse
import dolfinx
import logging
import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc

import commons, constants, configs


markers = commons.SurfaceMarkers()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run simulation..')
    parser.add_argument('--grid_size', help='Nx-Ny-Nz', required=True)
    parser.add_argument('--data_dir', help='directory with mesh files. output will be saved here', required=True, type=str)
    parser.add_argument("--voltage", nargs='?', const=1, default=1)
    parser.add_argument("--eps", help='fraction of area at left current collector that is in contact',
                        nargs='?', const=1, default=0.05, type=np.double)

    args = parser.parse_args()
    data_dir = args.data_dir
    voltage = args.voltage
    scaling = configs.get_configs()['VOXEL_SCALING']
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])
    loglevel = configs.get_configs()['LOGGING']['level']
    comm = MPI.COMM_WORLD
    rank = comm.rank
    start_time = timeit.default_timer()

    eps = args.eps
    grid_size = args.grid_size
    FORMAT = f'%(asctime)s: %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(f'{data_dir}')
    logger.setLevel(loglevel)
    Lx, Ly, Lz = [int(v) for v in grid_size.split("-")]
    Lx = Lx * scale_x
    Ly = Ly * scale_y
    Lz = Lz * scale_z
    tetr_mesh_path = os.path.join(data_dir, 'tetr.xdmf')
    tria_mesh_path = os.path.join(data_dir, 'tria.xdmf')
    output_current_path = os.path.join(data_dir, f'current.xdmf')
    output_potential_path = os.path.join(data_dir, 'potential.xdmf')

    left_cc_marker = markers.left_cc
    right_cc_marker = markers.right_cc
    insulated_marker = markers.insulated

    logger.debug("Loading tetrahedra (dim = 3) mesh..")
    with dolfinx.io.XDMFFile(comm, tetr_mesh_path, "r") as infile3:
        domain = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.none, 'Grid')
        ct = infile3.read_meshtags(domain, name="Grid")
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)

    with dolfinx.io.XDMFFile(comm, tria_mesh_path, "r") as infile3:
        ft = infile3.read_meshtags(domain, name="Grid")
    
    surf_meshtags = dolfinx.mesh.meshtags(domain, 2, ft.indices, ft.values)
    n = ufl.FacetNormal(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=surf_meshtags)

    # Dirichlet BCs
    V = dolfinx.fem.functionspace(domain, ("Lagrange", 2))
    u0 = dolfinx.fem.Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(voltage)

    u1 = dolfinx.fem.Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0.0)
    
    x0facet = ft.find(markers.left_cc)
    x1facet = ft.find(markers.right_cc)
    insulated_facet = ft.find(markers.insulated)
    surf_meshtags = dolfinx.mesh.meshtags(domain, 2, ft.indices, ft.values)

    x0bc = dolfinx.fem.dirichletbc(u0, dolfinx.fem.locate_dofs_topological(V, 2, x0facet))
    x1bc = dolfinx.fem.dirichletbc(u1, dolfinx.fem.locate_dofs_topological(V, 2, x1facet))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    # bulk conductivity [S.m-1]
    kappa_0 = dolfinx.fem.Constant(domain, PETSc.ScalarType(constants.KAPPA0))
    f = dolfinx.fem.Constant(domain, PETSc.ScalarType(0.0))
    g = dolfinx.fem.Constant(domain, PETSc.ScalarType(0.0))

    a = ufl.inner(kappa_0 * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ds(markers.insulated)

    options = {
               "ksp_type": "gmres",
               "pc_type": "hypre",
               "ksp_rtol": 1.0e-12
               }

    model = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[x0bc, x1bc], petsc_options=options)

    logger.debug('Solving problem..')
    uh = model.solve()
    
    # Save solution in XDMF format
    with dolfinx.io.XDMFFile(comm, output_potential_path, "w") as outfile:
        outfile.write_mesh(domain)
        outfile.write_function(uh)

    # # Update ghost entries and plot
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    # Post-processing: Compute derivatives
    grad_u = ufl.grad(uh)

    W = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
    current_expr = dolfinx.fem.Expression(kappa_0 * ufl.sqrt(ufl.inner(grad_u, grad_u)), W.element.interpolation_points())
    current_h = dolfinx.fem.Function(W)
    current_h.interpolate(current_expr)

    with dolfinx.io.XDMFFile(comm, output_current_path, "w") as file:
        file.write_mesh(domain)
        file.write_function(current_h)

    insulated_area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds))
    area_left_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds(markers.left_cc)))
    area_right_cc = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ds(markers.right_cc)))
    i_left_cc = (1/area_left_cc) * dolfinx.fem.assemble_scalar(dolfinx.fem.form(kappa_0 * ufl.sqrt(ufl.inner(grad_u, grad_u)) * ds(markers.left_cc)))
    i_right_cc = (1/area_right_cc) * dolfinx.fem.assemble_scalar(dolfinx.fem.form(kappa_0 * ufl.sqrt(ufl.inner(grad_u, grad_u)) * ds(markers.right_cc)))
    i_insulated = (1/insulated_area) * dolfinx.fem.assemble_scalar(dolfinx.fem.form(kappa_0 * ufl.sqrt(ufl.inner(grad_u, grad_u)) * ds(markers.insulated)))
    volume = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain)))
    solution_trace_norm = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(ufl.grad(uh), n) ** 2 * ds)) ** 0.5
    avg_solution_trace_norm = solution_trace_norm / insulated_area
    deviation_in_current = np.around(100 * 2 * np.abs(area_left_cc * i_left_cc - area_right_cc * i_right_cc) / (area_left_cc * i_left_cc + area_right_cc * i_right_cc), 2)
    logger.info("**************************RESULTS-SUMMARY******************************************")
    logger.info(f"Contact Area @ left cc [sq. um]                 : {area_left_cc:.4e}")
    logger.info(f"Contact Area @ right cc [sq. um]                : {area_right_cc:.4e}")
    logger.info(f"Current density @ left cc                       : {i_left_cc:.4e}")
    logger.info(f"Current density @ right cc                      : {i_right_cc:.4e}")
    logger.info(f"Insulated Area [sq. um]                         : {insulated_area:.4e}")
    logger.info("Total Area [sq. um]                             : {:.4e}".format(area_left_cc + area_right_cc + insulated_area))
    logger.info(f"Electrolyte Volume [cu. um]                     : {volume:.4e}")
    logger.info("Electrolyte Volume Fraction                     : {:.2%}".format(volume/(Lx * Ly * Lz)))
    logger.info("Bulk conductivity [S.m-1]                       : {:.4e}".format(0.1))
    logger.info("Effective conductivity [S.m-1]                  : {:.4e}".format(Lz * area_left_cc * i_left_cc / (voltage * (Lx * Ly))))
    logger.info(f"Homogeneous Neumann BC trace                    : {solution_trace_norm:.2e}")
    logger.info(f"Area-averaged Homogeneous Neumann BC trace      : {avg_solution_trace_norm:.2e}")
    logger.info("Deviation in current at two current collectors  : {:.2f}%".format(deviation_in_current))
    logger.info(f"Time elapsed                                    : {int(timeit.default_timer() - start_time):3.5f}s")
    logger.info(f"Voltage                                         : {args.voltage}")
    logger.info("*************************END-OF-SUMMARY*******************************************")