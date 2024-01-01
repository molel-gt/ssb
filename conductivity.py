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
from dolfinx import cpp, fem, io, mesh
from dolfinx.fem import petsc
from dolfinx.io import VTXWriter
from mpi4py import MPI
from petsc4py import PETSc

import commons, configs, constants

markers = commons.SurfaceMarkers()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    parser.add_argument('--root_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage", nargs='?', const=1, default=1e-3)
    parser.add_argument("--Wa", help="Wagna number -> charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=np.nan)
    parser.add_argument("--scale", help="sx,sy,sz", nargs='?', const=1, default=None)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='VOXEL_SCALING', type=str)
    parser.add_argument("--compute_distribution", help="compute current distribution stats", nargs='?', const=1, default=False, type=bool)
    parser.add_argument("--compute_grad_distribution", help="compute current distribution stats", nargs='?', const=1,
                        default=False, type=bool)
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="conductivity")

    args = parser.parse_args()
    data_dir = os.path.join(f'{args.root_folder}')
    voltage = args.voltage
    comm = MPI.COMM_WORLD
    rank = comm.rank
    start_time = timeit.default_timer()
    if args.scale is None:
        scaling = configs.get_configs()[args.scaling]
        scale_x = float(scaling['x'])
        scale_y = float(scaling['y'])
        scale_z = float(scaling['z'])
    else:
        scale_x, scale_y, scale_z = [float(vv) for vv in args.scale.split(',')]
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
    stats_path = os.path.join(data_dir, 'cdf.csv')
    frequency_path = os.path.join(data_dir, 'frequency.csv')
    grad_cd_path = os.path.join(data_dir, 'cdf_grad_cd.csv')
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
        meshtags = mesh.meshtags(domain, 2, ft.indices, ft.values)
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
        meshtags = mesh.meshtags(domain, domain.topology.dim - 1, ft_indices, ft_values)

    # Dirichlet BCs
    V = fem.functionspace(domain, ("Lagrange", 2))
    u0 = fem.Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(voltage)

    u1 = fem.Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0.0)

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

    model = petsc.LinearProblem(a, L, bcs=[left_bc, right_bc], petsc_options=options)
    logger.debug('Solving problem..')
    uh = model.solve()
    
    # Save solution in XDMF format
    # with io.XDMFFile(comm, output_potential_path, "w") as outfile:
    #     outfile.write_mesh(domain)
    #     outfile.write_function(uh)
    with VTXWriter(comm, output_potential_path, [uh], engine="BP4") as vtx:
        vtx.write(0.0)

    # # Update ghost entries and plot
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    logger.debug("Post-process calculations")
    grad_u = ufl.grad(uh)
    # W = fem.VectorFunctionSpace(domain, ("Lagrange", 1))
    # e = element("CG", "tetrahedron", 1, shape=(3,))
    W = fem.functionspace(domain, ("CG", 1, (3,)))
    current_expr = fem.Expression(-kappa * grad_u, W.element.interpolation_points())
    current_h = fem.Function(W)
    tol_fun = fem.Function(V)
    tol_fun_left = fem.Function(V)
    tol_fun_right = fem.Function(V)
    current_h.interpolate(current_expr)

    # with io.XDMFFile(comm, output_current_path, "w") as file:
    #     file.write_mesh(domain)
    #     file.write_function(current_h)
    with VTXWriter(comm, output_current_path, [current_h], engine="BP4") as vtx:
        vtx.write(0.0)

    logger.debug("Post-process Results Summary")
    insulated_area = domain.comm.reduce(fem.assemble_scalar(fem.form(1 * ds(markers.insulated))), op=MPI.SUM, root=0)
    area_left_cc = domain.comm.reduce(fem.assemble_scalar(fem.form(1 * ds(markers.left_cc))), op=MPI.SUM, root=0)
    area_right_cc = domain.comm.reduce(fem.assemble_scalar(fem.form(1 * ds(markers.right_cc))), op=MPI.SUM, root=0)
    I_left_cc = domain.comm.reduce(fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.left_cc))), op=MPI.SUM, root=0)
    I_right_cc = domain.comm.reduce(fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.right_cc))), op=MPI.SUM, root=0)
    I_insulated = domain.comm.reduce(fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds)), op=MPI.SUM, root=0)
    volume = domain.comm.reduce(fem.assemble_scalar(fem.form(1 * ufl.dx(domain))), op=MPI.SUM, root=0)

    # if args.compute_distribution:
        # logger.debug("Cumulative distribution lines of current density at terminals")
        # cd_lims = defaultdict(lambda : [0, 25])
        # cd_lims.update(
        #     {
        #         1: [0, 60],
        #         5: [0, 25],
        #         15: [0, 25],
        #         30: [0, 25],
        #         50: [0, 25],
        #         100: [0, 25],
        #         200: [0, 25],
        #     }
        # )
        # min_cd, max_cd = cd_lims[int(dimensions.split("-")[-1])]
        # cd_space = np.linspace(min_cd, max_cd, num=10000)
        # cdf_values = []
        # freq_values = []
        # EPS = 1e-30

        # def frequency_condition(values, vleft, vright):
        #     tol_fun_left.interpolate(lambda x: vleft * (x[0] + EPS) / (x[0] + EPS))
        #     tol_fun_right.interpolate(lambda x: vright * (x[0] + EPS) / (x[0] + EPS))
        #     return ufl.conditional(ufl.ge(values, tol_fun_left), 1, 0) * ufl.conditional(ufl.lt(values, tol_fun_right), 1, 0)

        # def check_condition(values, tol):
        #     tol_fun.interpolate(lambda x: tol * (x[0] + EPS) / (x[0] + EPS))
        #     return ufl.conditional(ufl.le(values, tol_fun), 1, 0)

        # for v in cd_space:
        #     lpvalue = domain.comm.reduce(fem.assemble_scalar(fem.form(check_condition(np.abs(ufl.inner(current_h, n)), v) * ds(markers.left_cc))) / area_left_cc, op=MPI.SUM, root=0)
        #     rpvalue = domain.comm.reduce(fem.assemble_scalar(fem.form(check_condition(np.abs(ufl.inner(current_h, n)), v) * ds(markers.right_cc))) / area_right_cc, op=MPI.SUM, root=0)
        #     cdf_values.append({'i [A/m2]': v, "p_left": lpvalue, "p_right": rpvalue})
        # for i, vleft in enumerate(list(cd_space)[:-1]):
        #     vright = cd_space[i+1]
        #     freql = domain.comm.reduce(
        #         fem.assemble_scalar(fem.form(frequency_condition(np.abs(ufl.inner(current_h, n)), vleft, vright) * ds(markers.left_cc))) / area_left_cc,
        #         op=MPI.SUM,
        #         root=0
        #     )
        #     freqr = domain.comm.reduce(
        #         fem.assemble_scalar(fem.form(frequency_condition(np.abs(ufl.inner(current_h, n)), vleft, vright) * ds(
        #             markers.right_cc))) / area_right_cc,
        #         op=MPI.SUM,
        #         root=0
        #     )
        #     freq_values.append({"vleft [A/m2]": vleft, "vright [A/m2]": vright, "freql": freql, "freqr": freqr})
        # if domain.comm.rank == 0:
        #     with open(stats_path, 'w') as fp:
        #         writer = csv.DictWriter(fp, fieldnames=['i [A/m2]', 'p_left', 'p_right'])
        #         writer.writeheader()
        #         for row in cdf_values:
        #             writer.writerow(row)
        #     with open(frequency_path, "w") as fp:
        #         writer = csv.DictWriter(fp, fieldnames=["vleft [A/m2]", "vright [A/m2]", "freql", "freqr"])
        #         writer.writeheader()
        #         for row in freq_values:
        #             writer.writerow(row)
        # logger.debug(f"Wrote cdf stats in {stats_path}")
        # if args.compute_grad_distribution:
        #     logger.debug(f"Cumulative distribution lines of derivative of current density at terminals")
        #     grad2 = ufl.sqrt(
        #         ufl.inner(
        #             ufl.grad(ufl.inner(current_h, n)),
        #             ufl.grad(ufl.inner(current_h, n))
        #         )
        #     )
        #     grad_cd_space = [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5] + list(np.linspace(1e-5 + 1e-6, 1e-4, num=100)) + [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
        #     grad_cd_cdf_values = []
        #     for v in grad_cd_space:
        #         lpvalue = fem.assemble_scalar(fem.form(check_condition(grad2, v) * ds(markers.left_cc))) / area_left_cc
        #         grad_cd_cdf_values.append({'i [A/m2]': v, "p_left": lpvalue, "p_right": "-"})
        #     with open(grad_cd_path, 'w') as fp:
        #         writer = csv.DictWriter(fp, fieldnames=['i [A/m2]', 'p_left', 'p_right'])
        #         writer.writeheader()
        #         for row in grad_cd_cdf_values:
        #             writer.writerow(row)
        #     logger.debug(f"Wrote cdf stats in {grad_cd_path}")
    if domain.comm.rank == 0:
        i_right_cc = I_right_cc / area_right_cc
        i_left_cc = I_left_cc / area_left_cc
        i_insulated = I_insulated / insulated_area
        volume_fraction = volume / (Lx * Ly * Lz)
        total_area = area_left_cc + area_right_cc + insulated_area
        error = (max([I_left_cc ** 2, I_right_cc ** 2]) / min([I_left_cc ** 2, I_right_cc ** 2])) ** 0.5

        simulation_metadata = {
            "Wagner number": args.Wa,
            "Contact area fraction at left electrode": area_left_cc / (Lx * Ly),
            "Contact area fraction at left electrode": area_right_cc / (Lx * Ly),
            "Contact area at left electrode [sq. m]": area_left_cc,
            "Contact area at right electrode [sq. m]": area_right_cc,
            "Average current density at active area of left electrode [A.m-2]": i_left_cc,
            "Average current density at active area of right electrode [A.m-2]": i_right_cc,
            "Dimensions Lx-Ly-Lz (unscaled)": args.dimensions,
            "Scaling for dimensions x,y,z to meters": args.scaling,
            "Bulk conductivity [S.m-1]": constants.KAPPA0,
            "Effective conductivity [S.m-1]": kappa_eff,
            "Current density at insulated area [A.m-2]": i_insulated,
            "Area-averaged Homogeneous Neumann BC trace": avg_solution_trace_norm,
            "Max electrode current over min electrode current (error)": error,
            "Simulation time (seconds)": f"{int(timeit.default_timer() - start_time):3.5f}",
            "Voltage drop [V]": args.voltage,
            "Insulated area [sq. m]": insulated_area,
            "Electrolyte volume fraction": volume_fraction,
            "Electrolyte volume [cu. m]": volume,
        }
        with open(simulation_metafile, "w", encoding='utf-8') as f:
            json.dump(simulation_metadata, f, ensure_ascii=False, indent=4)

        logger.info(f"Time elapsed                                    : {int(timeit.default_timer() - start_time):3.5f}s")
