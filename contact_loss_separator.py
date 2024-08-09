#!/usr/bin/env python3
import csv
import json
import os
import pickle
import timeit

import argparse
import logging
import matplotlib.pyplot as plt
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
from ufl import grad, inner

import commons, configs, constants, utils

markers = commons.Markers()

faraday_constant = 96485  # [C/mol]
R = 8.314  # [J/K/mol]
T = 298  # [K]
dtype = PETSc.ScalarType


def get_chunk(rank, size, n_points):
        chunk_size = int(np.ceil(n_points / size))
        if rank + 1 == size:
            return int(chunk_size) * rank, n_points
        else:
            return int(chunk_size) * rank, int(chunk_size * (rank+1)) + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="contact_loss_lma")
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1e-3, type=float)
    parser.add_argument("--Wa", help="Wagna number: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=np.nan, type=float)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='CONTACT_LOSS_SCALING', type=str)
    parser.add_argument("--galvanostatic", help="specify current of 10 A/m^2", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--compute_distribution", help="compute current distribution stats", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    voltage = args.voltage
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_time = timeit.default_timer()
    scaling = configs.get_configs()[args.scaling]
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])
    loglevel = configs.get_configs()['LOGGING']['level']
    dimensions = utils.extract_dimensions_from_meshfolder(args.mesh_folder)
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    formatter = logging.Formatter(f'%(levelname)s:%(asctime)s:{args.mesh_folder}:%(message)s')
    fh = logging.FileHandler(os.path.basename(__file__).replace(".py", ".log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.debug(args.mesh_folder)
    os.environ["XDG_CACHE_HOME"] = "/path/to/case3"

    Lx, Ly, Lz = [float(v) for v in dimensions.split("-")]
    Lx = Lx * scale_x
    Ly = Ly * scale_y
    Lz = Lz * scale_z
    results_dir = os.path.join(args.mesh_folder, f"{args.Wa}")
    utils.make_dir_if_missing(results_dir)
    output_meshfile_path = os.path.join(args.mesh_folder, 'mesh.msh')
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
    V = fem.functionspace(domain, ("CG", 2))
    
    n = ufl.FacetNormal(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    dx = ufl.Measure("dx", domain=domain, subdomain_data=ct)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # bulk conductivity [S.m-1]
    kappa = fem.Constant(domain, dtype(constants.KAPPA0))
    f = fem.Constant(domain, dtype(0.0))
    g = fem.Constant(domain, dtype(0.0))
    curr_converged = False
    curr_cd = 0
    target_cd = 10  # A/m2

    A0 = Lx * Ly

    if not args.galvanostatic:
        left_dofs = fem.locate_dofs_topological(V, 2, left_boundary)
        right_dofs = fem.locate_dofs_topological(V, 2, right_boundary)
        left_bc = fem.dirichletbc(dtype(0), left_dofs, V)
        right_bc = fem.dirichletbc(dtype(voltage), right_dofs, V)

        a = inner(kappa * grad(u), grad(v)) * dx
        L = inner(f, v) * dx + inner(g, v) * ds(markers.insulated)
        logger.debug(f'Solving problem..')
        # problem = petsc.NonlinearProblem(F, u, bcs=[left_bc, right_bc])
        # solver = petsc_nls.NewtonSolver(comm, problem)
        problem = petsc.LinearProblem(a, L, bcs=[left_bc, right_bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()
        # solver.convergence_criterion = "residual"
        # solver.maximum_iterations = 100
        # solver.atol = np.finfo(float).eps
        # solver.rtol = np.finfo(float).eps * 10

        # ksp = solver.krylov_solver
        # opts = PETSc.Options()
        # option_prefix = ksp.getOptionsPrefix()
        # opts[f"{option_prefix}ksp_type"] = "gmres"
        # opts[f"{option_prefix}pc_type"] = "hypre"
        # ksp.setFromOptions()
        # n_iters, converged = solver.solve(u)
        # logger.info(f"Converged in {n_iters} iterations")
        # u.x.scatter_forward()

    max_iters = 100
    its = 0

    while args.galvanostatic and not curr_converged and its < max_iters:
        its += 1
        left_dofs = fem.locate_dofs_topological(V, 2, left_boundary)
        right_dofs = fem.locate_dofs_topological(V, 2, right_boundary)
        left_bc = fem.dirichletbc(dtype(0), left_dofs, V)
        right_bc = fem.dirichletbc(dtype(voltage), right_dofs, V)

        F = inner(kappa * grad(u), grad(v)) * dx
        F -= inner(f, v) * dx + inner(g, v) * ds(markers.insulated)
        logger.debug(f'Iteration {its}: Solving problem..')
        problem = petsc.NonlinearProblem(F, u, bcs=[left_bc, right_bc])
        solver = petsc_nls.NewtonSolver(comm, problem)
        solver.convergence_criterion = "residual"
        solver.maximum_iterations = 100
        solver.atol = np.finfo(float).eps
        solver.rtol = np.finfo(float).eps * 10

        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "preonly"
        opts[f"{option_prefix}pc_type"] = "lu"
        ksp.setFromOptions()
        n_iters, converged = solver.solve(u)
        logger.info(f"Converged in {n_iters} iterations")
        u.x.scatter_forward()
        curr_cd = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(inner(-kappa * ufl.grad(u), n)) * ds(markers.right))), op=MPI.SUM) / A0
        logger.info(f"Iteration: {its}, current: {curr_cd}, target: {target_cd}")
        if np.isclose(curr_cd, target_cd, atol=0.01):
            curr_converged = True
        voltage *= target_cd / curr_cd

    with VTXWriter(comm, output_potential_path, [uh], engine="BP5") as vtx:
        vtx.write(0.0)

    logger.debug("Post-process calculations")
    W = fem.functionspace(domain, ("CG", 1, (3,)))
    current_expr = fem.Expression(-kappa * ufl.grad(uh), W.element.interpolation_points())
    current_h = fem.Function(W, name='current_density')
    tol_fun = fem.Function(V)
    tol_fun_left = fem.Function(V)
    tol_fun_right = fem.Function(V)
    current_h.interpolate(current_expr)

    with VTXWriter(comm, output_current_path, [current_h], engine="BP5") as vtx:
        vtx.write(0.0)

    logger.debug("Post-process Results Summary")
    insulated_area = domain.comm.allreduce(fem.assemble_scalar(fem.form(1 * ds(markers.insulated))), op=MPI.SUM)
    area_left_cc = domain.comm.allreduce(fem.assemble_scalar(fem.form(1 * ds(markers.left))), op=MPI.SUM)
    area_right_cc = domain.comm.allreduce(fem.assemble_scalar(fem.form(1 * ds(markers.right))), op=MPI.SUM)
    I_left_cc = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(current_h, n) * ds(markers.left))), op=MPI.SUM)
    I_right_cc = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(current_h, n) * ds(markers.right))), op=MPI.SUM)
    I_insulated = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(inner(current_h, n)) * ds(markers.insulated))), op=MPI.SUM)
    volume = domain.comm.allreduce(fem.assemble_scalar(fem.form(1 * dx(domain))), op=MPI.SUM)

    u_avg_left = domain.comm.allreduce(fem.assemble_scalar(fem.form(uh * ds(markers.left))), op=MPI.SUM) / area_left_cc
    u_avg_right = domain.comm.allreduce(fem.assemble_scalar(fem.form(uh * ds(markers.right))), op=MPI.SUM) / area_right_cc

    u_var_left = domain.comm.allreduce(fem.assemble_scalar(fem.form((1 / area_left_cc) * (uh - u_avg_left) ** 2 * ds(markers.left))))
    u_var_right = domain.comm.allreduce(fem.assemble_scalar(fem.form((1 / area_right_cc) * (uh - u_avg_right) ** 2 * ds(markers.right))))

    i_right_cc = I_right_cc / area_right_cc
    i_left_cc = I_left_cc / area_left_cc
    i_insulated = I_insulated / insulated_area
    var_left = domain.comm.allreduce(fem.assemble_scalar(fem.form((1 / area_left_cc) * (inner(current_h, n) - i_left_cc) ** 2 * ds(markers.left))))
    var_right = domain.comm.allreduce(fem.assemble_scalar(fem.form((1 / area_right_cc) * (inner(current_h, n) - i_right_cc) ** 2 * ds(markers.right))))
    volume_fraction = volume / (Lx * Ly * Lz)
    total_area = area_left_cc + area_right_cc + insulated_area
    error = max([np.abs(I_left_cc), np.abs(I_right_cc)]) / min([np.abs(I_left_cc), np.abs(I_right_cc)])
    kappa_eff = abs(I_left_cc) / A0 * Lz / voltage
    insulated_ratio = I_insulated / min(abs(I_left_cc), abs(I_right_cc))
    EPS = 1e-30
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
        num_points = 10000
        cd_space = np.linspace(min_cd, max_cd, num=num_points)
        cdf_values = []
        freq_values = [{}] * (num_points - 1)

        def frequency_condition(values, vleft, vright):
            tol_fun_left.interpolate(lambda x: vleft * (x[0] + EPS) / (x[0] + EPS))
            tol_fun_right.interpolate(lambda x: vright * (x[0] + EPS) / (x[0] + EPS))
            return ufl.conditional(ufl.ge(values, tol_fun_left), 1, 0) * ufl.conditional(ufl.lt(values, tol_fun_right), 1, 0)

        c_size = get_chunk(rank, size, num_points)

        for idx in range(c_size[0], c_size[1]-1):
            vleft = cd_space[idx]
            vright = cd_space[idx+1]
            freql = fem.assemble_scalar(fem.form(frequency_condition(np.abs(ufl.inner(current_h, n)), vleft, vright) * ds(markers.left)))
            freqr = fem.assemble_scalar(fem.form(frequency_condition(np.abs(ufl.inner(current_h, n)), vleft, vright) * ds(markers.right)))
            freq_values[idx] = {"vleft [A/m2]": vleft, "vright [A/m2]": vright, "freql": freql / A0, "freqr": freqr / A0}
        if domain.comm.rank == 0:
            with open(frequency_path, "w") as fp:
                writer = csv.DictWriter(fp, fieldnames=["vleft [A/m2]", "vright [A/m2]", "freql", "freqr"])
                writer.writeheader()
                for row in freq_values:
                    writer.writerow(row)
        logger.debug(f"Wrote frequency stats in {frequency_path}")
        def less_than_zero(val):
            if val < np.finfo(float).eps:
                return 1
            return 0
    tol_fun_right.interpolate(lambda x: x[0] + EPS)
    area_zero_curr = comm.allreduce(fem.assemble_scalar(fem.form(ufl.conditional(ufl.lt(np.abs(ufl.inner(current_h, n)), tol_fun_right), 1, 0) * ds(markers.right))), op=MPI.SUM)
    if domain.comm.rank == 0:
        logger.debug("Writing summary information..")
        simulation_metadata = {
            "Wagner number": args.Wa,
            "Contact area fraction at left electrode": f"{area_left_cc / (Lx * Ly):.4e}",
            "Contact area fraction at right electrode": f"{area_right_cc / (Lx * Ly):.4e}",
            "Contact area at left electrode [sq. m]": f"{area_left_cc:.4e}",
            "Contact area at right electrode [sq. m]": f"{area_right_cc:.4e}",
            "Insulated area [sq. m]": f"{insulated_area:.4e}",
            "Average current density at active area of left electrode [A.m-2]": f"{np.abs(i_left_cc):.4e}",
            "Average current density at active area of right electrode [A.m-2]": f"{np.abs(i_right_cc):.4e}",
            "Average potential at active area of left electrode [A.m-2]": f"{np.abs(u_avg_left):.4e}",
            "Average potential at active area of right electrode [A.m-2]": f"{np.abs(u_avg_right):.4e}",
            "STDEV current density left electrode [A/m2]": f"{np.sqrt(var_left):.4e}",
            "STDEV current density right electrode [A/m2]": f"{np.sqrt(var_right):.4e}",
            "STDEV potential left electrode [A/m2]": f"{np.sqrt(u_var_left):.4e}",
            "STDEV potential right electrode [A/m2]": f"{np.sqrt(u_var_right):.4e}",
            "Average current density at insulated area [A.m-2]": f"{i_insulated:.4e}",
            "Current at active area of left electrode [A]": f"{np.abs(I_left_cc):.4e}",
            "Current at active area of right electrode [A]": f"{np.abs(I_right_cc):.4e}",
            "Current at insulated area [A]": f"{np.abs(I_insulated):.4e}",
            "Dimensions Lx-Ly-Lz (unscaled)": dimensions,
            "Scaling for dimensions x,y,z to meters": args.scaling,
            "Bulk conductivity [S.m-1]": constants.KAPPA0,
            "Effective conductivity [S.m-1]": f"{kappa_eff:.4e}",
            "Max electrode current over min electrode current (error)": error,
            "Insulated current over min electrode current (error)": insulated_ratio,
            "Simulation time (seconds)": f"{int(timeit.default_timer() - start_time):,}",
            "Voltage drop [V]": voltage,
            "Electrolyte volume fraction": f"{volume_fraction:.4f}",
            "Electrolyte volume [cu. m]": f"{volume:.4e}",
            "Fraction less than zero at right electrode": area_zero_curr / A0,
            "Total resistance [Ω.cm2]": voltage / (np.abs(I_right_cc) / (A0 * 1e4)),
        }
        with open(simulation_metafile, "w", encoding='utf-8') as f:
            json.dump(simulation_metadata, f, ensure_ascii=False, indent=4)

        logger.info(f"Time elapsed                                    : {int(timeit.default_timer() - start_time):,}s")
