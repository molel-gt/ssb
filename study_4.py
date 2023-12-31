#!/usr/bin/env python3

import os
import json
import timeit

import argparse
import dolfinx
import logging
import numpy as np
import ufl

from dolfinx import cpp, fem, io, mesh, nls, plot
from mpi4py import MPI
from petsc4py import PETSc

import commons, constants, configs, study_4_utils


markers = commons.SurfaceMarkers()

# model parameters
KAPPA = 1e-1  # [S/m]
faraday_const = 96485  # [C/mol]
i0 = 1e1  # [A/m^2]
R = 8.314  # [J/K/mol]
T = 298  # [K]
z = 1
voltage = 0  # [V]
i_sup = 1e-1  # [A/m^2]
phi_m = 0  # [V]
U_therm = 0  # [V]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run simulation..')
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    parser.add_argument("--voltage", nargs='?', const=1, default=1)
    parser.add_argument("--eps", help='fraction of area at left current collector that is in contact',
                        nargs='?', const=1, default=0.05, type=np.double)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='STUDY4_VOXEL_SCALING', type=str)
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="study_4")
    parser.add_argument("--max_resolution", help="maximum resolution", nargs='?', const=1, default=1)

    args = parser.parse_args()

    voltage = args.voltage
    scaling = [float(configs.get_configs()[args.scaling][k]) for k in ['x', 'y', 'z']]
    max_resolution = args.max_resolution
    data_dir = study_4_utils.meshfile_subfolder_path(eps=args.eps, name_of_study=args.name_of_study, dimensions=args.dimensions, max_resolution=args.max_resolution)
    loglevel = configs.get_configs()['LOGGING']['level']
    comm = MPI.COMM_WORLD
    rank = comm.rank
    start_time = timeit.default_timer()

    eps = args.eps
    FORMAT = f'%(asctime)s: %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(f'{data_dir}')
    logger.setLevel(loglevel)
    Lx, Ly, Lz = [int(v) * scaling[idx] for idx, v in enumerate(args.dimensions.split("-"))]
    xsection_area = Lx * Ly
    Wa = KAPPA * R * T / (Lz * faraday_const * i0)
    tetr_mesh_path = os.path.join(data_dir, 'tetr.xdmf')
    tria_mesh_path = os.path.join(data_dir, 'tria.xdmf')
    output_current_path = os.path.join(data_dir, f'current.xdmf')
    output_potential_path = os.path.join(data_dir, 'potential.xdmf')
    simulation_metafile = os.path.join(data_dir, "simulation.json")

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
    
    meshtags = dolfinx.mesh.meshtags(domain, 2, ft.indices, ft.values)
    n = ufl.FacetNormal(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=meshtags)

    # Dirichlet BCs
    V = fem.FunctionSpace(domain, ("Lagrange", 2))
    u0 = fem.Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(voltage)

    u1 = fem.Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0.0)
    
    x0facet = ft.find(markers.left_cc)
    x1facet = ft.find(markers.right_cc)
    insulated_facet = ft.find(markers.insulated)
    meshtags = dolfinx.mesh.meshtags(domain, 2, ft.indices, ft.values)

    x0bc = fem.dirichletbc(u0, fem.locate_dofs_topological(V, 2, x0facet))
    x1bc = fem.dirichletbc(u1, fem.locate_dofs_topological(V, 2, x1facet))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    # bulk conductivity [S.m-1]
    kappa = fem.Constant(domain, PETSc.ScalarType(KAPPA))
    f = fem.Constant(domain, PETSc.ScalarType(0))
    g = fem.Constant(domain, PETSc.ScalarType(0))

    F = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    F -= ufl.inner(f, v) * ufl.dx
    bcs = [x0bc, x1bc]
    F += ufl.inner(g, v) * ds(markers.insulated)
    # s = fem.Constant(domain, PETSc.ScalarType(U_therm))
    r = fem.Constant(domain, PETSc.ScalarType(i0 * z * faraday_const / (R * T)))
    g_1 = fem.Constant(domain, PETSc.ScalarType(i_sup))
    # F += ufl.inner(g_1, v) * ds(markers.right_cc)
    # F += r * ufl.inner(phi_m - u - U_therm, v) * ds(markers.left_cc)
    options = {
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-12,
    }
    a = ufl.lhs(F)
    L = ufl.rhs(F)

    model = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options=options)

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

    W = fem.VectorFunctionSpace(domain, ("Lagrange", 1))
    current_expr = fem.Expression(-kappa * grad_u, W.element.interpolation_points())
    current_h = fem.Function(W)
    current_h.interpolate(current_expr)

    with dolfinx.io.XDMFFile(comm, output_current_path, "w") as file:
        file.write_mesh(domain)
        file.write_function(current_h)

    insulated_area = fem.assemble_scalar(fem.form(1 * ds(markers.insulated)))
    area_left_cc = fem.assemble_scalar(fem.form(1 * ds(markers.left_cc)))
    area_right_cc = fem.assemble_scalar(fem.form(1 * ds(markers.right_cc)))
    I_left_cc = abs(fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.left_cc))))
    i_left_cc = I_left_cc / area_left_cc
    I_right_cc = abs(fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.right_cc))))
    i_right_cc = I_right_cc / area_right_cc
    I_insulated = fem.assemble_scalar(fem.form(ufl.inner(current_h, n) * ds(markers.insulated)))
    i_insulated = I_insulated / insulated_area
    volume = fem.assemble_scalar(fem.form(1 * ufl.dx(domain)))
    avg_solution_trace_norm = i_insulated

    error = max([I_left_cc, I_right_cc]) / min([I_left_cc, I_right_cc])
    kappa_eff = Lz * I_left_cc / (voltage * xsection_area)
    volume_fraction = volume / (xsection_area * Lz)
    simulation_metadata = {
        "Wagner number": Wa,
        "Contact area fraction at left electrode": args.eps,
        "Contact area at left electrode [sq. m]": area_left_cc,
        "Contact area at right electrode [sq. m]": area_right_cc,
        "Average current density at active area of left electrode": i_left_cc,
        "Average current density at active area of right electrode": i_right_cc,
        "Dimensions": args.dimensions,
        "Scaling for dimensions x,y,z to meters": args.scaling,
        "Bulk conductivity [S.m-1]": KAPPA,
        "Effective conductivity [S.m-1]": kappa_eff,
        "Current density at insulated area": i_insulated,
        "Area-averaged Homogeneous Neumann BC trace": avg_solution_trace_norm,
        "Max electrode current over min electrode current (error)": error,
        "Simulation time (seconds)": f"{int(timeit.default_timer() - start_time):3.5f}",
        "Voltage": args.voltage,
        "Insulated area [sq. m]": insulated_area,
        "Electrolyte volume fraction": volume_fraction,
        "Electrolyte volume [cu. m]": volume,
    }
    if rank == 0:
        with open(simulation_metafile, "w", encoding='utf-8') as f:
            json.dump(simulation_metadata, f, ensure_ascii=False, indent=4)
    logger.info("**************************RESULTS-SUMMARY******************************************")
    logger.info(f"Wa                                              : {Wa}")
    logger.info(f"Contact Area @ left cc [sq. m]                 : {area_left_cc:.4e}")
    logger.info(f"Contact Area @ right cc [sq. m]                : {area_right_cc:.4e}")
    logger.info(f"Current density @ left cc                       : {i_left_cc:.4e}")
    logger.info(f"Current density @ right cc                      : {i_right_cc:.4e}")
    logger.info(f"Insulated Area [sq. m]                         : {insulated_area:.4e}")
    logger.info("Total Area [sq. m]                             : {:.4e}".format(area_left_cc + area_right_cc + insulated_area))
    logger.info(f"Electrolyte Volume [cu. m]                     : {volume:.4e}")
    logger.info("Electrolyte Volume Fraction                     : {:.2%}".format(volume_fraction))
    logger.info("Bulk conductivity [S.m-1]                       : {:.4e}".format(KAPPA))
    logger.info(f"Effective conductivity [S.m-1]                  : {kappa_eff:.4e}")
    logger.info(f"Current density @ insulated                     : {i_insulated:.2e}")
    logger.info(f"Area-averaged Homogeneous Neumann BC trace      : {avg_solution_trace_norm:.2e}")
    logger.info(f"Error                                            : {error:.2f}")
    logger.info(f"Time elapsed                                    : {int(timeit.default_timer() - start_time):3.5f}s")
    logger.info(f"Voltage                                         : {args.voltage}")
    logger.info("*************************END-OF-SUMMARY*******************************************")