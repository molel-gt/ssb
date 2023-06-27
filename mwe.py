#!/usr/bin/env python3

import logging
import numpy as np
import ufl

from dolfinx import cpp, fem, io, mesh
from mpi4py import MPI
from petsc4py import PETSc


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(message)s')
    fh = logging.FileHandler('transport.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    comm = MPI.COMM_WORLD

    logger.debug("Loading tetrahedra (dim = 3) mesh..")
    domain = mesh.create_box(comm, [[0, 0, 0], [10, 10, 10]], [20, 20, 20], mesh.CellType.tetrahedron)
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
    # meshtags = mesh.meshtags(domain, 2, ft.indices, ft.values)
    # Dirichlet BCs
    V = fem.FunctionSpace(domain, ("Lagrange", 2))
    u0 = fem.Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(1)

    u1 = fem.Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(0)

    left_boundary = mesh.locate_entities_boundary(domain, dim=(domain.topology.dim - 1),
                                                  marker=lambda x: np.isclose(x[2], 0.0))
    right_boundary = mesh.locate_entities_boundary(domain, dim=(domain.topology.dim - 1),
                                                  marker=lambda x: np.isclose(x[2], 10.0))
    insulated_boundary = mesh.locate_entities_boundary(domain, dim=(domain.topology.dim - 1),
                                                   marker=lambda x: np.logical_not(np.isclose(x[2], 0), np.isclose(x[2], 10)))
    lmarker = 1
    rmarker = 2
    imarker = 3

    left_bc = fem.dirichletbc(u0, fem.locate_dofs_topological(V, 2, left_boundary))
    right_bc = fem.dirichletbc(u1, fem.locate_dofs_topological(V, 2, right_boundary))
    n = ufl.FacetNormal(domain)
    x = ufl.SpatialCoordinate(domain)
    # ds = ufl.Measure("ds", domain=domain, subdomain_data=meshtags)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # bulk conductivity [S.m-1]
    kappa = fem.Constant(domain, PETSc.ScalarType(0.1))
    f = fem.Constant(domain, PETSc.ScalarType(0.0))
    g = fem.Constant(domain, PETSc.ScalarType(0.0))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ufl.ds

    options = {
               "ksp_type": "gmres",
               "pc_type": "hypre",
               "ksp_rtol": 1.0e-12
               }

    model = fem.petsc.LinearProblem(a, L, bcs=[left_bc, right_bc], petsc_options=options)
    logger.debug('Solving problem..')
    uh = model.solve()
    
    # Save solution in XDMF format
    with io.XDMFFile(comm, "u.xdmf", "w") as outfile:
        outfile.write_mesh(domain)
        outfile.write_function(uh)

    # # Update ghost entries and plot
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    logger.debug("Post-process calculations")
    # Post-processing: Compute derivatives
    W = fem.VectorFunctionSpace(domain, ("Lagrange", 1))
    grad_expr = fem.Expression(-kappa * ufl.grad(uh), W.element.interpolation_points())
    grad_h = fem.Function(W)
    cdf_fun = fem.Function(W)
    cdf_fun2 = fem.Function(W)
    grad_h.interpolate(grad_expr)

    with io.XDMFFile(comm, "gradu.xdmf", "w") as file:
        file.write_mesh(domain)
        file.write_function(grad_h)

    logger.debug("Post-process Results Summary")
    # distribution at surfaces
    EPS = 1e-30
    def check_condition(v1, check_value=1):
        v2 = lambda x: check_value * (x[0] + EPS) / (x[0] + EPS)
        cdf_fun.interpolate(v2)
        return ufl.conditional(ufl.le(v1, cdf_fun), v1, cdf_fun)
    tol = 0.5
    new_v = fem.Expression(check_condition(ufl.inner(grad_h, n), tol), W.element.interpolation_points())
    cdf_fun2.interpolate(new_v)
    # lpvalue = fem.assemble_scalar(fem.form(ufl.inner(cdf_fun2, n) * ds(lmarker)))
    # rpvalue = fem.assemble_scalar(fem.form(ufl.inner(cdf_fun2, n) * ds(rmarker)))
    # logger.info((lpvalue, rpvalue))
