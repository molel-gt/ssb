#!/usr/bin/env python3

import os
import time

import argparse
import logging
import numpy as np
import ufl

from dolfinx import cpp, fem, io, mesh, nls, plot
from mpi4py import MPI
from petsc4py import PETSc

import commons, configs, constants, geometry, utils

markers = commons.SurfaceMarkers()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Diffusivity')
    parser.add_argument('--grid_extents', help='Nx-Ny-Nz_Ox-Oy-Oz size_location', required=True)
    parser.add_argument('--root_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage", nargs='?', const=1, default=1e-3)
    parser.add_argument("--scale", help="sx,sy,sz", nargs='?', const=1, default='-1,-1,-1')
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='VOXEL_SCALING', type=str)
    args = parser.parse_args()
    data_dir = os.path.join(f'{args.root_folder}')
    loglevel = configs.get_configs()['LOGGING']['level']
    grid_extents = args.grid_extents
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    formatter = logging.Formatter(f'%(levelname)s:%(asctime)s:{grid_extents}:%(message)s')
    fh = logging.FileHandler('transport.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    comm = MPI.COMM_WORLD

    if args.scale == '-1,-1,-1':
        scaling = configs.get_configs()[args.scaling]
        scale_x = float(scaling['x'])
        scale_y = float(scaling['y'])
        scale_z = float(scaling['z'])
    else:
        scale_x, scale_y, scale_z = [float(vv) for vv in args.scale.split(',')]

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

    # potential problem
    Q = fem.FunctionSpace(domain, ("Lagrange", 2))

    u0 = fem.Function(Q)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(args.voltage)

    u1 = fem.Function(Q)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0.0)

    left_boundary = ft.find(markers.left_cc)
    right_boundary = ft.find(markers.right_cc)
    left_bc = fem.dirichletbc(u0, fem.locate_dofs_topological(Q, 2, left_boundary))
    right_bc = fem.dirichletbc(u1, fem.locate_dofs_topological(Q, 2, right_boundary))
    n = ufl.FacetNormal(domain)
    x = ufl.SpatialCoordinate(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=meshtags)

    # Define variational problem
    u = ufl.TrialFunction(Q)
    v = ufl.TestFunction(Q)

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

    logger.debug("Post-process calculations")
    grad_u = ufl.grad(uh)
    W = fem.VectorFunctionSpace(domain, ("Lagrange", 1))
    current_expr = fem.Expression(-kappa * grad_u, W.element.interpolation_points())
    current_h = fem.Function(W)
    tol_fun = fem.Function(Q)
    current_h.interpolate(current_expr)

    with io.XDMFFile(comm, output_current_path, "w") as file:
        file.write_mesh(domain)
        file.write_function(current_h)

    # concentration problem
    # bulk diffusivity
    D = fem.Constant(domain, PETSc.ScalarType(1e-10))
    f_farad = fem.Constant(domain, PETSc.ScalarType(96485))
    # surface flux conditions
    V = fem.FunctionSpace(domain, ("Lagrange", 2))
    dt = 1e-3  # [ms]
    theta = 0.5  # crank-nicholson time-stepping
    c = ufl.TrialFunction(V)
    c0 = ufl.TrialFunction(V)  # solution from previous converged step
    r = ufl.TestFunction(V)
    c.x.array[:] = 1e3  # [mol/m3]

    c_mid = (1.0 - theta) * c0 + theta * c
    g0 = fem.Constant(domain, PETSc.ScalarType(0.0))

    F = ufl.inner(c, r) * ufl.dx - ufl.inner(c0, r) * ufl.dx + dt * ufl.inner(D * ufl.grad(c_mid), ufl.grad(r)) * ufl.dx
    F -= ufl.inner(g0, r) * ds(markers.insulated)
    F -= ufl.inner(g0, r) * ds(markers.left_cc)
    F -= ufl.inner(ufl.inner((1 / D / f_farad) * current_h, n), r) * ds(markers.right_cc)

    # Create nonlinear problem and Newton solver
    problem = fem.petsc.NonlinearProblem(F, c)
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = np.sqrt(np.finfo(np.float64).eps) * 1e-2

    # We can customize the linear solver used inside the NewtonSolver by
    # modifying the PETSc options
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "gmres"
    opts[f"{option_prefix}pc_type"] = "hypre"
    ksp.setFromOptions()

    c0.x.array[:] = c.x.array
    # step in time
    t = 0.0
    T = 50 * dt
    while t < T:
        t += dt
        ret = solver.solve(c)
        print(f"Step {int(t / dt)}: num iterations: {ret[0]}")
        c0.x.array[:] = c.x.array
        file.write_function(c, t)
    file.close()
