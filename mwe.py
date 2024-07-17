import os

import dolfinx
import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
import ufl
import warnings

from dolfinx import cpp, default_scalar_type, fem, io, mesh, nls, plot
from dolfinx.fem import petsc
from dolfinx.io import VTXWriter
from dolfinx.nls import petsc as petsc_nls
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 dot, div, dx, ds, dS, grad, inner, grad, avg, jump)

import commons, geometry, utils

warnings.simplefilter('ignore')


def create_mesh(mesh, cell_type, prune_z=False):
    """
    """
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points,
                           cells={cell_type: cells},
                           cell_data={"name_to_read": [cell_data]}
                           )
    if prune_z:
        out_mesh.prune_z_0()

    return out_mesh

if __name__ == '__main__':
    # facet labels
    left = 0
    right = 2
    middle = 3
    insulated = 4

    # subdomain labels
    left_domain = 0
    right_domain = 1

    encoding = io.XDMFFile.Encoding.HDF5
    LX = 50e-6
    LY = 250e-6
    points = [
        (0, 0, 0),
        (0.5 * LX, 0, 0),
        (LX, 0, 0),
        (LX, LY, 0),
        (0.5 * LX, LY, 0),
        (0, LY, 0),
    ]
    gpoints = []
    lines = []

    gmsh.initialize()
    gmsh.model.add('full-cell')
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1 * micron)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5e-6)
    for idx, p in enumerate(points):
        gpoints.append(
            gmsh.model.occ.addPoint(*p)
        )
    gmsh.model.occ.synchronize()
    gmsh.model.occ.synchronize()
    for idx in range(0, len(points)-1):
        lines.append(
            gmsh.model.occ.addLine(gpoints[idx], gpoints[idx+1])
        )
    lines.append(
        gmsh.model.occ.addLine(gpoints[-1], gpoints[0])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints[1], gpoints[4])
    )

    gmsh.model.occ.synchronize()
    ltag = gmsh.model.addPhysicalGroup(1, [lines[-2]], left)
    gmsh.model.setPhysicalName(1, ltag, "left")
    rtag = gmsh.model.addPhysicalGroup(1, [lines[2]], right)
    gmsh.model.setPhysicalName(1, rtag, "right")
    evptag = gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [-1]], middle)
    gmsh.model.setPhysicalName(1, evptag, "middle")
    insultag = gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [0, 1, 3, 4]], insulated)
    gmsh.model.setPhysicalName(1, insultag, "insulated")
    gmsh.model.occ.synchronize()
    left_loop = gmsh.model.occ.addCurveLoop([lines[idx] for idx in [0, 6, 4, 5]])
    right_loop = gmsh.model.occ.addCurveLoop([lines[idx] for idx in [1, 2, 3, 6]])
    gmsh.model.occ.synchronize()
    left_phase = gmsh.model.occ.addPlaneSurface([left_loop])
    right_phase = gmsh.model.occ.addPlaneSurface([right_loop])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [left_phase], left_domain)
    gmsh.model.addPhysicalGroup(2, [right_phase], right_domain)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.write("mesh.msh")
    gmsh.finalize()

    mesh_2d = meshio.read("mesh.msh")
    tria_mesh = create_mesh(mesh_2d, "triangle")
    meshio.write("triangles.xdmf", tria_mesh)
    line_mesh = create_mesh(mesh_2d, "line")
    meshio.write("lines.xdmf", line_mesh)

    # simulation
    comm = MPI.COMM_WORLD
    with io.XDMFFile(comm, "triangles.xdmf", "r") as infile3:
        domain = infile3.read_mesh(cpp.mesh.GhostMode.none, 'Grid')
        ct = infile3.read_meshtags(domain, name="Grid")
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)

    ft_imap = domain.topology.index_map(fdim)
    num_facets = ft_imap.size_local + ft_imap.num_ghosts
    indices = np.arange(0, num_facets)
    values = np.zeros(indices.shape, dtype=np.intc)  # all facets are tagged with zero

    with io.XDMFFile(comm, "lines.xdmf", "r") as infile2:
        ft = infile2.read_meshtags(domain, name="Grid")

    values[ft.indices] = ft.values
    meshtags = mesh.meshtags(domain, fdim, indices, values)
    domaintags = mesh.meshtags(domain, domain.topology.dim, ct.indices, ct.values)

    dx = ufl.Measure("dx", domain=domain, subdomain_data=domaintags)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=meshtags)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=meshtags)
    V = fem.functionspace(domain, ("DG", 1))
    u = fem.Function(V)
    u.name = "potential"
    v = ufl.TestFunction(V)
    n = ufl.FacetNormal(domain)
    x = ufl.SpatialCoordinate(domain)

    h = ufl.CellDiameter(domain)
    h_avg = avg(h)

    f = fem.Constant(domain, PETSc.ScalarType(0))
    g = fem.Constant(domain, PETSc.ScalarType(0))
    voltage = 1000e-3
    u_left = fem.Function(V)
    with u_left.vector.localForm() as u0_loc:
        u0_loc.set(0)
    u_right = fem.Function(V)
    with u_right.vector.localForm() as u1_loc:
        u1_loc.set(voltage)

    # domain-varying parameter
    Κ = fem.Function(V)
    cells_left = ct.find(left_domain)
    cells_right = ct.find(right_domain)
    Κ.x.array[cells_left] = np.full_like(cells_left, 0.1, dtype=default_scalar_type)
    Κ.x.array[cells_right] = np.full_like(cells_right, 1, dtype=default_scalar_type)

    # constant for kinetics with -K * inner(grad(u), n) = D0 * (jump(u, n) - U0)
    D0 = 4000
    U0 = ufl.as_vector((0.66, 0.66, 0.66))

    α = 100
    γ = 1e5

    F = inner(Κ * grad(u), grad(v)) * dx

    # Add DG/IP terms
    F += - inner(avg(grad(v)), jump(u, n)) * dS
    F += - inner(jump(v, n), avg(grad(u))) * dS
    F +=  γ / h_avg * inner(jump(v, n), jump(u, n)) * dS

    # Nitsche boundary terms
    F += - Κ * v * inner(grad(u), n) * ds(left)
    F += - Κ * v * inner(grad(u), n) * ds(right)
    F += - g * v * ds(insulated)
    F += - Κ * (u - u_left) * inner(grad(v), n) * ds(left)
    F += - Κ * (u - u_right) * inner(grad(v), n) * ds(right) 
    F += -h / α * u * inner(grad(v), n) * ds(insulated)
    F += α / h * (u - u_left) * v * ds(left) 
    F += α / h * (u - u_right) * v * ds(right)

    # Middle boundary
    F += + inner(avg(grad(v)), 1 / D0 * (Κ * grad(u))('-') - U0) * dS(middle)
    F += + inner(jump(v, n), avg(Κ * grad(u))) * dS(middle)
    F += - γ / h_avg * inner(jump(v, n), 1 / D0 * (Κ * grad(u))('+') - U0) * dS(middle)

    # Source term
    F += -f * v * dx

    # problem and solution
    problem = petsc.NonlinearProblem(F, u)
    solver = petsc_nls.NewtonSolver(comm, problem)
    solver.convergence_criterion = "residual"
    solver.maximum_iterations = 5

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    ksp.setFromOptions()
    n_iters, converged = solver.solve(u)

    with VTXWriter(comm, "potential.bp", [u], engine="BP4") as vtx:
        vtx.write(0.0)