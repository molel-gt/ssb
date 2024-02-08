#!/usr/bin/env python3
import csv
import json
import os
import timeit

import argparse
import logging
import numpy as np
import ufl

from basix.ufl import element
from collections import defaultdict

from dolfinx import cpp, default_scalar_type, fem, io, mesh
from dolfinx.fem import petsc
from dolfinx.io import VTXWriter
from mpi4py import MPI
from petsc4py import PETSc

import commons, configs, constants

markers = commons.SurfaceMarkers()

faraday_constant = 96485  # [C/mol]
R = 8.314  # [J/K/mol]
T = 298  # [K]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="conductivity")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1e-3)
    parser.add_argument("--Wa", help="Wagna number: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=0)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='CONTACT_LOSS_SCALING', type=str)
    parser.add_argument("--compute_distribution", help="compute current distribution stats", nargs='?', const=1, default=False, type=bool)

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

    Lx, Ly, Lz = [float(v) for v in dimensions.split("-")]
    Lx = Lx * scale_x
    Ly = Ly * scale_y
    Lz = Lz * scale_z
    tetr_mesh_path = os.path.join(data_dir, 'tetr.xdmf')
    tria_mesh_path = os.path.join(data_dir, 'tria.xdmf')
    output_current_path = os.path.join(data_dir, 'current.bp')
    output_potential_path = os.path.join(data_dir, 'potential.bp')
    frequency_path = os.path.join(data_dir, 'frequency.csv')
    simulation_metafile = os.path.join(data_dir, 'simulation.json')

    left_cc_marker = markers.left_cc
    right_cc_marker = markers.right_cc
    insulated_marker = markers.insulated

    logger.debug("Loading tetrahedra (dim = 3) mesh..")
    with io.XDMFFile(comm, tetr_mesh_path, "r") as infile3:
        domain = infile3.read_mesh(cpp.mesh.GhostMode.none, 'Grid')
        ct = infile3.read_meshtags(domain, name="Grid")
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
    try:
        logger.debug("Attempting to load xmdf file for triangle mesh")
        with io.XDMFFile(comm, tria_mesh_path, "r") as infile2:
            ft = infile2.read_meshtags(domain, name="Grid")
        left_boundary = ft.find(markers.left_cc)
        right_boundary = ft.find(markers.right_cc)
    except RuntimeError as e:
        logger.error("Missing xdmf file for triangle mesh!")
        facets = mesh.locate_entities_boundary(domain, dim=domain.topology.dim - 1,
                                               marker=lambda x: np.isfinite(x[2]))
        facets_l0 = mesh.locate_entities_boundary(domain, dim=domain.topology.dim - 1,
                                               marker=lambda x: np.isclose(x[2], 0))
        facets_lz = mesh.locate_entities_boundary(domain, dim=domain.topology.dim - 1,
                                               marker=lambda x: np.isclose(x[2], Lz))
        all_indices = set(tuple([val for val in facets]))
        l0_indices = set(tuple([val for val in facets_l0]))
        lz_indices = set(tuple([val for val in facets_lz]))
        insulator_indices = all_indices.difference(l0_indices | lz_indices)
        ft_indices = np.asarray(list(l0_indices) + list(lz_indices) + list(insulator_indices), dtype=np.int32)
        ft_values = np.asarray([markers.left_cc] * len(l0_indices) + [markers.right_cc] * len(lz_indices) + [markers.insulated] * len(insulator_indices), dtype=np.int32)
        left_boundary = facets_l0
        right_boundary = facets_lz
        ft = mesh.meshtags(domain, domain.topology.dim - 1, ft_indices, ft_values)

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
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # bulk conductivity [S.m-1]
    kappa = fem.Constant(domain, PETSc.ScalarType(constants.KAPPA0))
    f = fem.Constant(domain, PETSc.ScalarType(0.0))
    g = fem.Constant(domain, PETSc.ScalarType(0.0))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ds(markers.insulated) + ufl.inner(i_exchange * faraday_constant * (u - 0) / (constants.KAPPA0 * R * T), v) * ds(markers.left_cc)

    options = {
               "ksp_type": "gmres",
               "pc_type": "hypre",
               "ksp_rtol": 1.0e-12
               }

    model = petsc.LinearProblem(a, L, bcs=[right_bc], petsc_options=options)
    logger.debug('Solving problem..')
    uh = model.solve()

    with VTXWriter(comm, output_potential_path, [uh], engine="BP4") as vtx:
        vtx.write(0.0)

    logger.debug("Post-process calculations")
    W = fem.functionspace(domain, ("CG", 1, (3,)))
    current_expr = fem.Expression(-kappa * ufl.grad(uh), W.element.interpolation_points())
    current_h = fem.Function(W)
    tol_fun = fem.Function(V)
    tol_fun_left = fem.Function(V)
    tol_fun_right = fem.Function(V)
    current_h.interpolate(current_expr)

    with VTXWriter(comm, output_current_path, [current_h], engine="BP4") as vtx:
        vtx.write(0.0)

    logger.debug("Post-process Results Summary")
    insulated_area = domain.comm.allreduce(fem.assemble_scalar(fem.form(1 * ds(markers.insulated))), op=MPI.SUM)
    area_left_cc = domain.comm.allreduce(fem.assemble_scalar(fem.form(1 * ds(markers.left_cc))), op=MPI.SUM)
    area_right_cc = domain.comm.allreduce(fem.assemble_scalar(fem.form(1 * ds(markers.right_cc))), op=MPI.SUM)
    I_left_cc = domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.left_cc))), op=MPI.SUM)
    I_right_cc = domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.right_cc))), op=MPI.SUM)
    I_insulated = domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds)), op=MPI.SUM)
    volume = domain.comm.allreduce(fem.assemble_scalar(fem.form(1 * ufl.dx(domain))), op=MPI.SUM)
    A0 = Lx * Ly

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
        min_cd, max_cd = cd_lims[int(dimensions.split("-")[-1])]
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
                fem.assemble_scalar(fem.form(frequency_condition(np.abs(ufl.inner(current_h, n)), vleft, vright) * ds(markers.left_cc))),
                op=MPI.SUM
            )
            freqr = domain.comm.allreduce(
                fem.assemble_scalar(fem.form(frequency_condition(np.abs(ufl.inner(current_h, n)), vleft, vright) * ds(
                    markers.right_cc))),
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
        i_right_cc = I_right_cc / area_right_cc
        i_left_cc = I_left_cc / area_left_cc
        i_insulated = I_insulated / insulated_area
        volume_fraction = volume / (Lx * Ly * Lz)
        total_area = area_left_cc + area_right_cc + insulated_area
        error = max([np.abs(I_left_cc), np.abs(I_right_cc)]) / min([np.abs(I_left_cc), np.abs(I_right_cc)])
        kappa_eff = Lz * abs(I_left_cc) / (voltage * (Lx * Ly))

        simulation_metadata = {
            "Wagner number": args.Wa,
            "Contact area fraction at left electrode": f"{area_left_cc / (Lx * Ly):.4f}",
            "Contact area fraction at right electrode": f"{area_right_cc / (Lx * Ly):.4f}",
            "Contact area at left electrode [sq. m]": f"{area_left_cc:.4e}",
            "Contact area at right electrode [sq. m]": f"{area_right_cc:.4e}",
            "Insulated area [sq. m]": f"{insulated_area:.4e}",
            "Average current density at active area of left electrode [A.m-2]": f"{np.abs(i_left_cc):.4e}",
            "Average current density at active area of right electrode [A.m-2]": f"{np.abs(i_right_cc):.4e}",
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
        }
        with open(simulation_metafile, "w", encoding='utf-8') as f:
            json.dump(simulation_metadata, f, ensure_ascii=False, indent=4)

        logger.info(f"Time elapsed                                    : {int(timeit.default_timer() - start_time):3.5f}s")
