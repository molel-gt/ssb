#!/usr/bin/env python3
import csv
import os
import timeit

import argparse
import logging
import numpy as np
import ufl

from dolfinx import cpp, fem, io, mesh
from mpi4py import MPI
from petsc4py import PETSc

import commons, configs, constants

markers = commons.SurfaceMarkers()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument('--grid_extents', help='Nx-Ny-Nz_Ox-Oy-Oz size_location', required=True)
    parser.add_argument('--root_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage", nargs='?', const=1, default=1)
    parser.add_argument("--scale", help="sx,sy,sz", nargs='?', const=1, default='-1,-1,-1')
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='VOXEL_SCALING', type=str)
    args = parser.parse_args()
    data_dir = os.path.join(f'{args.root_folder}')
    voltage = args.voltage
    comm = MPI.COMM_WORLD
    rank = comm.rank
    start_time = timeit.default_timer()
    if args.scale == '-1,-1,-1':
        scaling = configs.get_configs()[args.scaling]
        scale_x = float(scaling['x'])
        scale_y = float(scaling['y'])
        scale_z = float(scaling['z'])
    else:
        scale_x, scale_y, scale_z = [float(vv) for vv in args.scale.split(',')]
    loglevel = configs.get_configs()['LOGGING']['level']

    grid_extents = args.grid_extents
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(message)s')
    fh = logging.FileHandler('transport.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    Lx, Ly, Lz = [float(v) - 1 for v in grid_extents.split("_")[0].split("-")]
    Lx = Lx * scale_x
    Ly = Ly * scale_y
    Lz = Lz * scale_z
    tetr_mesh_path = os.path.join(data_dir, 'tetr.xdmf')
    tria_mesh_path = os.path.join(data_dir, 'tria.xdmf')
    output_current_path = os.path.join(data_dir, 'current.xdmf')
    output_potential_path = os.path.join(data_dir, 'potential.xdmf')

    left_cc_marker = markers.left_cc
    right_cc_marker = markers.right_cc
    insulated_marker = markers.insulated

    logger.debug("Loading tetrahedra (dim = 3) mesh..")
    with io.XDMFFile(comm, tetr_mesh_path, "r") as infile3:
        domain = infile3.read_mesh(cpp.mesh.GhostMode.none, 'Grid')
        ct = infile3.read_meshtags(domain, name="Grid")
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
    with io.XDMFFile(comm, tria_mesh_path, "r") as infile2:
        ft = infile2.read_meshtags(domain, name="Grid")
    meshtags = mesh.meshtags(domain, 2, ft.indices, ft.values)
    # Dirichlet BCs
    V = fem.FunctionSpace(domain, ("Lagrange", 2))
    u0 = fem.Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(voltage)

    u1 = fem.Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0.0)

    left_boundary = ft.find(markers.left_cc)
    right_boundary = ft.find(markers.right_cc)
    left_bc = fem.dirichletbc(u0, fem.locate_dofs_topological(V, 2, left_boundary))
    right_bc = fem.dirichletbc(u1, fem.locate_dofs_topological(V, 2, right_boundary))
    n = ufl.FacetNormal(domain)
    # x = ufl.SpatialCoordinate(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=meshtags)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # bulk conductivity [S.m-1]
    kappa = fem.Constant(domain, PETSc.ScalarType(constants.KAPPA0))
    f = fem.Constant(domain, PETSc.ScalarType(0.0))
    g = fem.Constant(domain, PETSc.ScalarType(0.0))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ds(markers.insulated)

    options = {
               "ksp_type": "gmres",
               "pc_type": "hypre",
               "ksp_rtol": 1.0e-12
               }

    model = fem.petsc.LinearProblem(a, L, bcs=[left_bc, right_bc], petsc_options=options)
    logger.debug('Solving problem..')
    uh = model.solve()
    
    # Save solution in XDMF format
    with io.XDMFFile(comm, output_potential_path, "w") as outfile:
        outfile.write_mesh(domain)
        outfile.write_function(uh)

    # # Update ghost entries and plot
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    # Post-processing: Compute derivatives
    grad_u = ufl.grad(uh)

    W = fem.VectorFunctionSpace(domain, ("Lagrange", 1))
    current_expr = fem.Expression(-kappa * grad_u, W.element.interpolation_points())
    current_h = fem.Function(W)
    new_fun = fem.Function(W)
    current_h.interpolate(current_expr)

    with io.XDMFFile(comm, output_current_path, "w") as file:
        file.write_mesh(domain)
        file.write_function(current_h)

    insulated_area = fem.assemble_scalar(fem.form(1 * ds(markers.insulated)))
    area_left_cc = fem.assemble_scalar(fem.form(1 * ds(markers.left_cc)))
    area_right_cc = fem.assemble_scalar(fem.form(1 * ds(markers.right_cc)))
    I_left_cc = fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.left_cc)))
    i_left_cc = I_left_cc / area_left_cc
    I_right_cc = fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.right_cc)))
    i_right_cc = I_right_cc / area_right_cc
    I_insulated = fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds))
    i_insulated = I_insulated / insulated_area
    volume = fem.assemble_scalar(fem.form(1 * ufl.dx(domain)))
    total_area = area_left_cc + area_right_cc + insulated_area
    error = 100 * 2 * abs(abs(I_left_cc) - abs(I_right_cc)) / (abs(I_left_cc) + abs(I_right_cc))
    # distribution at terminals
    min_cd = np.min(current_h.x.array)
    max_cd = np.max(current_h.x.array)
    cd_space = np.linspace(min_cd, max_cd, num=1000)
    cdf_values = []
    # def value_is_less_than(value_1, value_2):
    #     return ufl.conditional(ufl.le(value_1, value_2), 1, 0)
    # check_arr = []
    # for value in np.asarray(ufl.inner(current_h, n)):
    #     check_arr.append(value_is_less_than(value, 0))
    # print(check_arr)
    # print(ufl.le(current_h.x.array, 0.5))
    # print(x[0], x[1], x[2], x.ufl_shape)
    # new_fun.x.array[:] = 0.5
    dummy = np.ones((2575446, ))
    # print(new_fun.x.array)
    new_fun.interpolate(lambda x: (dummy, dummy, dummy))
    new_express = fem.Expression(ufl.conditional(ufl.le(current_h, 0.5), 1, 0), W.element.interpolation_points())
    new_fun.interpolate(new_express)
    print(new_fun.x.array)
    print(ufl.inner(ufl.conditional(ufl.le(current_h.x.array, 0.5), 1, 0), n))
    print(fem.assemble_scalar(fem.form(ufl.conditional(ufl.le(ufl.inner(current_h.x.array, n), v), 1, 0) * ds(markers.left_cc))))
    for v in cd_space:
        lpvalue = fem.assemble_scalar(fem.form(ufl.conditional(ufl.le(ufl.inner(current_h, n), v), 1, 0) * ds(markers.left_cc))) / area_left_cc
        rpvalue = fem.assemble_scalar(fem.form(ufl.conditional(ufl.le(ufl.inner(current_h, n), v), 1, 0) * ds(markers.right_cc))) / area_right_cc
        cdf_values.append({'i [A/m2]': v, "p_left": lpvalue, "p_right": rpvalue})
    stats_path = os.path.join(data_dir, 'cdf.csv')
    with open(stats_path, 'w') as fp:
        writer = csv.DictWriter(fp, fieldnames=['i [A/m2]', 'p_left', 'p_right'])
        writer.writeheader()
        for row in cdf_values:
            writer.writerow(row)
    logger.debug(f"Wrote cdf stats in {stats_path}")
    logger.info("**************************RESULTS-SUMMARY******************************************")
    logger.info(f"Contact Area @ left cc [sq. um]                 : {area_left_cc*1e12:.4e}")
    logger.info(f"Contact Area @ right cc [sq. um]                : {area_right_cc*1e12:.4e}")
    logger.info(f"Current density @ left cc                       : {i_left_cc:.4e}")
    logger.info(f"Current density @ right cc                      : {i_right_cc:.4e}")
    logger.info(f"Insulated Area [sq. um]                         : {insulated_area*1e12:.4e}")
    logger.info(f"Total Area [sq. um]                             : {total_area*1e12:.4e}")
    logger.info(f"Electrolyte Volume [cu. um]                     : {volume*1e18:.4e}")
    logger.info("Electrolyte Volume Fraction                     : {:.2%}".format(volume / (Lx * Ly * Lz)))
    logger.info(f"Bulk conductivity [S.m-1]                       : {constants.KAPPA0:.4e}")
    logger.info("Effective conductivity [S.m-1]                  : {:.4e}".format(Lz * abs(I_left_cc) / (voltage * (Lx * Ly))))
    logger.info(f"Insulated Current [A] : {I_insulated:.2e}")
    logger.info(f"Deviation in current at two current collectors  : {error:.2f}%")
    logger.info(f"Voltage                                         : {args.voltage}")
    logger.info(f"Time elapsed                                    : {int(timeit.default_timer() - start_time):3.5f}s")
    logger.info("*************************END-OF-SUMMARY*******************************************")
