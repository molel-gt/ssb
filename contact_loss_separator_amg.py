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

from dolfinx import cpp, default_scalar_type, fem, io, la, mesh
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


def build_nullspace(V):
    """Build PETSc nullspace for 3D elasticity"""

    # Create vectors that will span the nullspace
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=dtype) for i in range(6)]
    b = [b.array for b in basis]

    # Get dof indices for each subspace (x, y and z dofs)
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]

    # Set the three translational rigid body modes
    for i in range(3):
        b[i][dofs[i]] = 1.0

    # Set the three rotational rigid body modes
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    b[3][dofs[0]] = -x1
    b[3][dofs[1]] = x0
    b[4][dofs[0]] = x2
    b[4][dofs[2]] = -x0
    b[5][dofs[2]] = x1
    b[5][dofs[1]] = -x2

    la.orthonormalize(basis)

    basis_petsc = [
        PETSc.Vec().createWithArray(x[: bs * length0], bsize=3, comm=V.mesh.comm)  # type: ignore
        for x in b
    ]
    return PETSc.NullSpace().create(vectors=basis_petsc)  # type: ignore


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
    rank = comm.rank
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
    results_dir = os.path.join(args.mesh_folder, f"{args.Wa}", "amg")
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

        a = fem.form(inner(kappa * grad(u), grad(v)) * dx)
        L = fem.form(inner(f, v) * dx + inner(g, v) * ds(markers.insulated))
        bcs = [left_bc, right_bc]
        A = petsc.assemble_matrix(a, bcs=bcs)
        A.assemble()
        b = petsc.assemble_vector(L)
        petsc.apply_lifting(b, [a], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)

        # ns = build_nullspace(V)
        # A.setNearNullSpace(ns)
        A.setOption(PETSc.Mat.Option.SPD, True)

        # Set solver options
        opts = PETSc.Options()  # type: ignore
        opts["ksp_type"] = "cg"
        opts["ksp_rtol"] = 1.0e-9
        opts["pc_type"] = "gamg"

        # Use Chebyshev smoothing for multigrid
        opts["mg_levels_ksp_type"] = "chebyshev"
        opts["mg_levels_pc_type"] = "jacobi"

        # Improve estimate of eigenvalues for Chebyshev smoothing
        opts["mg_levels_ksp_chebyshev_esteig_steps"] = 10

        # Create PETSc Krylov solver and turn convergence monitoring on
        solver = PETSc.KSP().create(comm)
        solver.setFromOptions()

        # Set matrix operator
        solver.setOperators(A)
        uh = fem.Function(V)

        # Set a monitor, solve linear system, and display the solver
        # configuration
        solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
        solver.solve(b, uh.x.petsc_vec)
        solver.view()

        # Scatter forward the solution vector to update ghost values
        uh.x.scatter_forward()


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
    insulated_ratio = I_insulated / max(abs(I_left_cc), abs(I_right_cc))

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
            "Insulated current over max electrode current (error)": insulated_ratio,
            "Simulation time (seconds)": f"{int(timeit.default_timer() - start_time):,}",
            "Voltage drop [V]": voltage,
            "Electrolyte volume fraction": f"{volume_fraction:.4f}",
            "Electrolyte volume [cu. m]": f"{volume:.4e}",
            "Total resistance [Ω.cm2]": voltage / (np.abs(I_right_cc) / (A0 * 1e4)),
        }
        with open(simulation_metafile, "w", encoding='utf-8') as f:
            json.dump(simulation_metadata, f, ensure_ascii=False, indent=4)

        logger.info(f"Time elapsed                                    : {int(timeit.default_timer() - start_time):,}s")
