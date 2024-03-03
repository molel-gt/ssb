#!/usr/bin/env python3
import csv
import json
import os
import pickle
import timeit

import argparse
import logging
import numpy as np
import ufl

from basix.ufl import element
from collections import defaultdict

from dolfinx import cpp, default_scalar_type, fem, io, mesh
from dolfinx.fem import petsc
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import gmshio, VTXWriter
from dolfinx.nls import petsc as petsc_nls
from mpi4py import MPI
from petsc4py import PETSc

import commons, configs, constants, utils

markers = commons.Markers()

faraday_constant = 96485  # [C/mol]
R = 8.314  # [J/K/mol]
T = 298  # [K]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="conductivity")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1e-3)
    parser.add_argument("--Wa", help="Wagna number: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=np.nan, type=float)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='CONTACT_LOSS_SCALING', type=str)
    parser.add_argument("--compute_distribution", help="compute current distribution stats", default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    data_dir = os.path.join(f'{args.mesh_folder}')
    voltage = args.voltage
    comm = MPI.COMM_WORLD
    rank = comm.rank
    start_time = timeit.default_timer()
    scaling = configs.get_configs()[args.scaling]
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])
    loglevel = configs.get_configs()['LOGGING']['level']
    dimensions = args.dimensions
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    formatter = logging.Formatter(f'%(levelname)s:%(asctime)s:{data_dir}:{dimensions}:%(message)s')
    fh = logging.FileHandler(os.path.basename(__file__).replace(".py", ".log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.debug(args.mesh_folder)

    Lx, Ly, Lz = [float(v) for v in dimensions.split("-")]
    Lx = Lx * scale_x
    Ly = Ly * scale_y
    Lz = Lz * scale_z
    results_dir = os.path.join(data_dir, f"{args.Wa}")
    utils.make_dir_if_missing(results_dir)
    output_meshfile_path = os.path.join(data_dir, 'trial.msh')
    tetr_mesh_path = os.path.join(data_dir, 'tetr.xdmf')
    tria_mesh_path = os.path.join(data_dir, 'tria.xdmf')
    output_current_path = os.path.join(results_dir, 'current.bp')
    output_potential_path = os.path.join(results_dir, 'potential.bp')
    frequency_path = os.path.join(results_dir, 'frequency.csv')
    simulation_metafile = os.path.join(results_dir, 'simulation.json')
    left_values_path = os.path.join(results_dir, 'left_values')
    right_values_path = os.path.join(results_dir, 'right_values')

    left_cc_marker = markers.left
    right_cc_marker = markers.right
    insulated_marker = markers.insulated

    logger.debug("Loading mesh..")
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = gmshio.read_from_msh(output_meshfile_path, comm, partitioner=partitioner)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)
    # ft_imap = domain.topology.index_map(fdim)
    # num_facets = ft_imap.size_local + ft_imap.num_ghosts
    # indices = np.arange(0, num_facets)
    # values = np.zeros(indices.shape, dtype=np.intc)  # all facets are tagged with zero
    # values[ft.indices] = ft.values
    # ft = mesh.meshtags(domain, fdim, indices, values)
    # ct = mesh.meshtags(domain, domain.topology.dim, ct.indices, ct.values)
    left_boundary = ft.find(markers.left)
    right_boundary = ft.find(markers.right)
    logger.debug("done\n")

    # Dirichlet BCs
    V = fem.functionspace(domain, ("Lagrange", 2))
    u0 = fem.Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(voltage)

    i_exchange = constants.KAPPA0 * R * T / (Lz * args.Wa * faraday_constant)
    right_bc = fem.dirichletbc(u0, fem.locate_dofs_topological(V, 2, right_boundary))
    n = ufl.FacetNormal(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)

    # Define variational problem
    u = fem.Function(V)
    v = ufl.TestFunction(V)

    # bulk conductivity [S.m-1]
    kappa = fem.Constant(domain, PETSc.ScalarType(constants.KAPPA0))
    f = fem.Constant(domain, PETSc.ScalarType(0.0))
    g = fem.Constant(domain, PETSc.ScalarType(0.0))

    F = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    F -= ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ds(markers.insulated) + ufl.inner(i_exchange * faraday_constant * (u - 0) / (constants.KAPPA0 * R * T), v) * ds(markers.left)
    logger.debug('Solving problem..')
    problem = petsc.NonlinearProblem(F, u, bcs=[right_bc])
    solver = petsc_nls.NewtonSolver(comm, problem)
    solver.convergence_criterion = "residual"
    solver.maximum_iterations = 10

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "gmres"
    opts[f"{option_prefix}pc_type"] = "lu"
    ksp.setFromOptions()
    n_iters, converged = solver.solve(u)
    if not converged:
        logger.debug(f"Solver did not converge in {n_iters} iterations")
    else:
        logger.info(f"Converged in {n_iters} iterations")
    u.name = 'potential'

    with VTXWriter(comm, output_potential_path, [u], engine="BP4") as vtx:
        vtx.write(0.0)

    logger.debug("Post-process calculations")
    W = fem.functionspace(domain, ("CG", 1, (3,)))
    current_expr = fem.Expression(-kappa * ufl.grad(u), W.element.interpolation_points())
    current_h = fem.Function(W)
    tol_fun = fem.Function(V)
    tol_fun_left = fem.Function(V)
    tol_fun_right = fem.Function(V)
    current_h.interpolate(current_expr)

    with VTXWriter(comm, output_current_path, [current_h], engine="BP4") as vtx:
        vtx.write(0.0)

    logger.debug("Post-process Results Summary")
    insulated_area = domain.comm.allreduce(fem.assemble_scalar(fem.form(1 * ds(markers.insulated))), op=MPI.SUM)
    area_left_cc = domain.comm.allreduce(fem.assemble_scalar(fem.form(1 * ds(markers.left))), op=MPI.SUM)
    area_right_cc = domain.comm.allreduce(fem.assemble_scalar(fem.form(1 * ds(markers.right))), op=MPI.SUM)
    I_left_cc = domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.left))), op=MPI.SUM)
    I_right_cc = domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.right))), op=MPI.SUM)
    I_insulated = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(ufl.inner(current_h, n)) * ds)), op=MPI.SUM)
    volume = domain.comm.allreduce(fem.assemble_scalar(fem.form(1 * ufl.dx(domain))), op=MPI.SUM)
    A0 = Lx * Ly

    n_points = 250000
    tol = 1e-12
    bb_trees = bb_tree(domain, domain.topology.dim)
    x_coords = np.linspace(tol, Lx - tol, n_points)
    y_coords = np.linspace(tol, Ly - tol, n_points)
    z_coords_1 = np.zeros(n_points)  # left boundary
    z_coords_2 = np.ones(n_points) * Lz  # right boundary
    # left boundary
    points1 = np.zeros((3, n_points))
    points1[0] = x_coords
    points1[1] = y_coords
    points1[2] = z_coords_1
    u_values1 = []
    cells1 = []
    points_on_proc1 = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates1 = compute_collisions_points(bb_trees, points1.T)
    # Choose one of the cells that contains the point
    colliding_cells1 = compute_colliding_cells(domain, cell_candidates1, points1.T)
    for i, point in enumerate(points1.T):
        if len(colliding_cells1.links(i)) > 0:
            points_on_proc1.append(point)
            cells1.append(colliding_cells1.links(i)[0])
    points_on_proc1 = np.array(points_on_proc1, dtype=np.float64)
    u_values1 = current_h.eval(points_on_proc1, cells1)
    values_left = np.vstack((points_on_proc1, u_values1[:, 2]))
    dbfile_left = open(left_values_path, 'ab')
    pickle.dump(values_left, dbfile_left)
    dbfile_left.close()
    print(u_values1.shape, points_on_proc1.shape, values_left.shape)

    # right boundary
    points2 = np.zeros((3, n_points))
    points2[0] = x_coords
    points2[1] = y_coords
    points2[2] = z_coords_2
    u_values2 = []
    cells2 = []
    points_on_proc2 = []

    # Find cells whose bounding-box collide with the the points
    cell_candidates2 = compute_collisions_points(bb_trees, points2.T)
    # Choose one of the cells that contains the point
    colliding_cells2 = compute_colliding_cells(domain, cell_candidates2, points2.T)
    for i, point in enumerate(points2.T):
        if len(colliding_cells2.links(i)) > 0:
            points_on_proc2.append(point)
            cells2.append(colliding_cells2.links(i)[0])
    points_on_proc2 = np.array(points_on_proc2, dtype=np.float64)
    u_values2 = current_h.eval(points_on_proc2, cells2)
    values_right = np.vstack((points_on_proc2, u_values2[:, 2]))
    dbfile_right = open(right_values_path, 'ab')
    pickle.dump(values_right, dbfile_right)
    dbfile_right.close()

    if args.compute_distribution:
        logger.debug("Cumulative distribution lines of current density at terminals")
        cd_lims = defaultdict(lambda : [0, 25])
        cd_lims.update(
            {
                1: [0, 60],
                5: [0, 25],
                15: [0, 25],
                30: [0, 25],
                50: [0, 25],
                100: [0, 25],
                200: [0, 25],
            }
        )

        if np.isnan(args.Wa):
            min_cd, max_cd = cd_lims[int(dimensions.split("-")[-1])]
        else:
            min_cd = 0
            max_cd = min(abs(I_left_cc/A0) * 5, cd_lims[int(dimensions.split("-")[-1])][-1])
        cd_space = np.linspace(min_cd, max_cd, num=10000)
        cdf_values = []
        freq_values = []
        EPS = 1e-30

        def frequency_condition(values, vleft, vright):
            tol_fun_left.interpolate(lambda x: vleft * (x[0] + EPS) / (x[0] + EPS))
            tol_fun_right.interpolate(lambda x: vright * (x[0] + EPS) / (x[0] + EPS))
            return ufl.conditional(ufl.ge(values, tol_fun_left), 1, 0) * ufl.conditional(ufl.lt(values, tol_fun_right), 1, 0)

        for i, vleft in enumerate(list(cd_space)[:-1]):
            vright = cd_space[i+1]
            freql = domain.comm.allreduce(
                fem.assemble_scalar(fem.form(frequency_condition(np.abs(ufl.inner(current_h, n)), vleft, vright) * ds(markers.left))),
                op=MPI.SUM
            )
            freqr = domain.comm.allreduce(
                fem.assemble_scalar(fem.form(frequency_condition(np.abs(ufl.inner(current_h, n)), vleft, vright) * ds(
                    markers.right))),
                op=MPI.SUM
            )
            freq_values.append({"vleft [A/m2]": vleft, "vright [A/m2]": vright, "freql": freql / A0, "freqr": freqr / A0})
        if domain.comm.rank == 0:
            with open(frequency_path, "w") as fp:
                writer = csv.DictWriter(fp, fieldnames=["vleft [A/m2]", "vright [A/m2]", "freql", "freqr"])
                writer.writeheader()
                for row in freq_values:
                    writer.writerow(row)
        logger.debug(f"Wrote frequency stats in {frequency_path}")
    if domain.comm.rank == 0:
        logger.debug("Generating summary information..")
        i_right_cc = I_right_cc / area_right_cc
        i_left_cc = I_left_cc / area_left_cc
        i_insulated = I_insulated / insulated_area
        var_left = domain.comm.allreduce(fem.assemble_scalar(fem.form((1 / area_left_cc) * (ufl.inner(current_h, n) - i_left_cc) ** 2 * ds(markers.left))))
        var_right = domain.comm.allreduce(fem.assemble_scalar(fem.form((1 / area_right_cc) * (ufl.inner(current_h, n) - i_right_cc) ** 2 * ds(markers.right))))
        volume_fraction = volume / (Lx * Ly * Lz)
        total_area = area_left_cc + area_right_cc + insulated_area
        error = max([np.abs(I_left_cc), np.abs(I_right_cc)]) / min([np.abs(I_left_cc), np.abs(I_right_cc)])
        kappa_eff = abs(I_left_cc) / A0 * Lz / voltage
        logger.debug("Finished generating summary.")

        simulation_metadata = {
            "Wagner number": args.Wa,
            "Contact area fraction at left electrode": f"{area_left_cc / (Lx * Ly):.4f}",
            "Contact area fraction at right electrode": f"{area_right_cc / (Lx * Ly):.4f}",
            "Contact area at left electrode [sq. m]": f"{area_left_cc:.4e}",
            "Contact area at right electrode [sq. m]": f"{area_right_cc:.4e}",
            "Insulated area [sq. m]": f"{insulated_area:.4e}",
            "Average current density at active area of left electrode [A.m-2]": f"{np.abs(i_left_cc):.4e}",
            "Average current density at active area of right electrode [A.m-2]": f"{np.abs(i_right_cc):.4e}",
            "STDEV current density left electrode [A/m2]": f"{np.sqrt(var_left):.4e}",
            "STDEV current density right electrode [A/m2]": f"{np.sqrt(var_right):.4e}",
            "Average current density at insulated area [A.m-2]": f"{i_insulated:.4e}",
            "Current at active area of left electrode [A]": f"{np.abs(I_left_cc):.4e}",
            "Current at active area of right electrode [A]": f"{np.abs(I_right_cc):.4e}",
            "Current at insulated area [A]": f"{np.abs(I_insulated):.4e}",
            "Dimensions Lx-Ly-Lz (unscaled)": args.dimensions,
            "Scaling for dimensions x,y,z to meters": args.scaling,
            "Bulk conductivity [S.m-1]": constants.KAPPA0,
            "Effective conductivity [S.m-1]": f"{kappa_eff:.4f}",
            "Max electrode current over min electrode current (error)": error,
            "Simulation time (seconds)": f"{int(timeit.default_timer() - start_time):,}",
            "Voltage drop [V]": args.voltage,
            "Electrolyte volume fraction": f"{volume_fraction:.4f}",
            "Electrolyte volume [cu. m]": f"{volume:.4e}",
            "Total resistance [Ω.cm2]": args.voltage / (I_right_cc / (A0 * 1e4)),
        }
        with open(simulation_metafile, "w", encoding='utf-8') as f:
            json.dump(simulation_metadata, f, ensure_ascii=False, indent=4)

        logger.info(f"Time elapsed                                    : {int(timeit.default_timer() - start_time):3.5f}s")
