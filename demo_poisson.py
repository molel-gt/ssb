from contextlib import ExitStack

import dolfinx
import numpy as np
import ufl
from dolfinx import la
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_vector, dirichletbc, Expression,
                         form, Function, FunctionSpace, locate_dofs_topological, set_bc,
                         VectorFunctionSpace
                         )
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
from petsc4py import PETSc
from ufl import ds, dx, grad, inner


def build_nullspace(V):
    """Build PETSc nullspace for 3D elasticity"""

    # Create list of vectors for building nullspace
    index_map = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    ns = [la.create_petsc_vector(index_map, bs) for i in range(4)]
    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in ns]
        basis = [np.asarray(x) for x in vec_local]

        # Get dof indices for each subspace (x, and y dofs)
        dofs = [V.sub(i).dofmap.list.array for i in range(2)]

        # Build the two translational rigid body modes
        for i in range(2):
            basis[i][dofs[i]] = 1.0

        # Build the two rotational rigid body modes
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.array
        x0, x1 = x[dofs_block, 0], x[dofs_block, 1]
        basis[2][dofs[0]] = -x1
        basis[2][dofs[1]] = x0
        basis[3][dofs[0]] = -x0
        basis[3][dofs[1]] = x1

    # Orthonormalise the six vectors
    la.orthonormalize(ns)
    assert la.is_orthonormal(ns)

    return PETSc.NullSpace().create(vectors=ns)

with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as infile3:
    mesh = infile3.read_mesh(dolfinx.cpp.mesh.GhostMode.shared_facet, 'Grid')
print("done loading triangular mesh")

tdim = mesh.topology.dim
fdim = tdim - 2

V = VectorFunctionSpace(mesh, ("Lagrange", 1))

# Define boundary condition on x = 0 or x = 1
u0 = Function(V)
with u0.vector.localForm() as u0_loc:
    u0_loc.set(1.0)
u1 = Function(V)
with u1.vector.localForm() as u1_loc:
    u1_loc.set(0)
x0facets = locate_entities_boundary(mesh, fdim,
                                    lambda x: np.isclose(x[0], 0.0))
x1facets = locate_entities_boundary(mesh, fdim,
                                    lambda x: np.isclose(x[0], 10.0))
x0bc = dirichletbc(u0, locate_dofs_topological(V, fdim, x0facets))
x1bc = dirichletbc(u1, locate_dofs_topological(V, fdim, x1facets))

# zero-flux sides
# free_end_facets = np.sort(locate_entities_boundary(mesh, 1, lambda x: np.isclose(abs(x[1] - 5.0), 5)))
# mt = dolfinx.mesh.MeshTags(mesh, 1, free_end_facets, 1)
# ds = ufl.Measure("ds", subdomain_data=mt)

# Define variational problem
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
x = ufl.SpatialCoordinate(mesh)

f = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0.0, 0.0)))
g = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0.0, 0.0)))

a = form(inner(grad(u), grad(v)) * dx(x))
L = form(inner(f, v) * dx + inner(g, v) * ds)

A = assemble_matrix(a, bcs=[x0bc, x1bc])
A.assemble()

b = assemble_vector(L)
apply_lifting(b, [a], bcs=[[x0bc, x1bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [x0bc, x1bc])

null_space = build_nullspace(V)
A.setNearNullSpace(null_space)

# Set solver options
opts = PETSc.Options()
opts["ksp_type"] = "cgs"
# opts["ksp_rtol"] = 1.0e-05
opts["pc_type"] = "lu"
opts['pc_hypre_type'] = 'boomeramg'
opts["pc_hypre_boomeramg_max_iter"] = 1
opts["pc_hypre_boomeramg_cycle_type"] = "v"

# # # Use Chebyshev smoothing for multigrid
opts["mg_levels_ksp_type"] = "chebyshev"
opts["mg_levels_pc_type"] = "jacobi"

# # # Improve estimate of eigenvalues for Chebyshev smoothing
opts["mg_levels_esteig_ksp_type"] = "cg"
opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

# Create PETSc Krylov solver and turn convergence monitoring on
solver = PETSc.KSP().create(mesh.comm)
solver.setFromOptions()

# Set matrix operator
solver.setOperators(A)

uh = Function(V)

# Set a monitor, solve linear system, and dispay the solver configuration
# solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
solver.solve(b, uh.vector)
# solver.view()

uh.x.scatter_forward()

# Save solution in XDMF format
with XDMFFile(MPI.COMM_WORLD, "potential.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(uh)

# Post-processing: Compute derivatives
grad_u = ufl.sym(grad(uh)) #* ufl.Identity(len(uh))

W = FunctionSpace(mesh, ("Discontinuous Lagrange", 0))
current_expr = Expression(ufl.sqrt(inner(grad_u, grad_u)), W.element.interpolation_points)
current_h = Function(W)
current_h.interpolate(current_expr)

with XDMFFile(MPI.COMM_WORLD, "current.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(current_h)
