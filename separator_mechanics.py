#!/usr/bin/env python3
import argparse
import json
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coupling of stress and lithium metal/electrolyte active area fraction.')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="separator_mechanics")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1e-3)
    parser.add_argument("--Wa", help="Wagna number: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=np.inf)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='VOXEL_SCALING', type=str)

    args = parser.parse_args()
    data_dir = os.path.join(f'{args.mesh_folder}')

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

    u1 = fem.Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0.0)

    left_bc = fem.dirichletbc(u0, fem.locate_dofs_topological(V, 2, left_boundary))
    right_bc = fem.dirichletbc(u1, fem.locate_dofs_topological(V, 2, right_boundary))
    n = ufl.FacetNormal(domain)
    # x = ufl.SpatialCoordinate(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)

    # Define variational problem
    u = ufl.TrialFunction(V)
    δu = ufl.TestFunction(V)

    # bulk conductivity [S.m-1]
    kappa = fem.Constant(domain, PETSc.ScalarType(constants.KAPPA0))
    f = fem.Constant(domain, PETSc.ScalarType(0.0))
    g = fem.Constant(domain, PETSc.ScalarType(0.0))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(δu)) * ufl.dx
    L = ufl.inner(f, δu) * ufl.dx + ufl.inner(g, δu) * ds(markers.insulated)

    options = {
               "ksp_type": "gmres",
               "pc_type": "hypre",
               "ksp_rtol": 1.0e-12
               }

    model = petsc.LinearProblem(a, L, bcs=[left_bc, right_bc], petsc_options=options)
    logger.debug('Solving problem..')
    uh = model.solve()

    with VTXWriter(comm, output_potential_path, [uh], engine="BP4") as vtx:
        vtx.write(0.0)

    logger.debug("Post-process calculations")
    W = fem.functionspace(domain, ("CG", 1, (3,)))
    current_expr = fem.Expression(-kappa * ufl.grad(uh), W.element.interpolation_points())
    current_h = fem.Function(W)
    current_h.interpolate(current_expr)

    with VTXWriter(comm, output_current_path, [current_h], engine="BP4") as vtx:
        vtx.write(0.0)