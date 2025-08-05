# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Stokes equations using Taylor-Hood elements
#
# This demo is implemented in {download}`demo_stokes.py`. It shows how
# to solve the Stokes problem using Taylor-Hood elements using different
# linear solvers.
#
# ## Equation and problem definition
#
# ### Strong formulation
#
# $$
# \begin{align}
#   - \nabla \cdot (\nabla u + p I) &= f \quad {\rm in} \ \Omega,\\
#   \nabla \cdot u &= 0 \quad {\rm in} \ \Omega.
# \end{align}
# $$
#
# with conditions on the boundary $\partial \Omega = \Gamma_{D} \cup
# \Gamma_{N}$ of the form:
#
# $$
# \begin{align}
#   u &= u_0 \quad {\rm on} \ \Gamma_{D},\\
#   \nabla u \cdot n + p n &= g \,   \quad\;\; {\rm on} \ \Gamma_{N}.
# \end{align}
# $$
#
# ```{note}
# The sign of the pressure has been changed from the usual
# definition. This is to generate have a symmetric system
# of equations.
# ```
#
# ### Weak formulation
#
# The weak formulation reads: find $(u, p) \in V \times Q$ such that
#
# $$
# a((u, p), (v, q)) = L((v, q)) \quad \forall  (v, q) \in V \times Q
# $$
#
# where
#
# $$
# \begin{align}
#   a((u, p), (v, q)) &:= \int_{\Omega} \nabla u \cdot \nabla v -
#            \nabla \cdot v \ p + \nabla \cdot u \ q \, {\rm d} x,
#   L((v, q)) &:= \int_{\Omega} f \cdot v \, {\rm d} x + \int_{\partial
#            \Omega_N} g \cdot v \, {\rm d} s.
# \end{align}
# $$
#
# ### Domain and boundary conditions
#
# We consider the lid-driven cavity problem with the following
# domain and boundary conditions:
#
# - $\Omega := [0,1]\times[0,1]$ (a unit square)
# - $\Gamma_D := \partial \Omega$
# - $u_0 := (1, 0)^\top$ at $x_1 = 1$ and $u_0 = (0, 0)^\top$ otherwise
# - $f := (0, 0)^\top$
#
#
# ## Implementation
#
# The Stokes problem using Taylor-Hood elements is solved using:
# 1. [Block preconditioner using PETSc MatNest and VecNest data
#    structures. Each 'block' is a standalone object.](#nested-matrix-solver)

from mpi4py import MPI

try:
    from petsc4py import PETSc

    import dolfinx

    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
except ModuleNotFoundError:
    print("This demo requires petsc4py.")
    exit(0)

import numpy as np

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem, io, la
from dolfinx.fem import (
    Constant,
    Function,
    dirichletbc,
    extract_function_spaces,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from ufl import div, dx, grad, inner

opts = PETSc.Options()
opts["mat_superlu_dist_iterrefine"] = True


# We create a {py:class}`Mesh <dolfinx.mesh.Mesh>`, define functions for
# locating geometrically subsets of the boundary, and define a function
# for the  velocity on the lid:

# +
# Create mesh
msh = create_rectangle(
    MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [32, 32], CellType.triangle
)


# Function to mark x = 0, x = 1 and y = 0
def noslip_boundary(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)


# Function to mark the lid (y = 1)
def lid(x):
    return np.isclose(x[1], 1.0)


# Lid velocity
def lid_velocity_expression(x):
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))


# -

# Two {py:class}`function spaces <dolfinx.fem.FunctionSpace>` are
# defined using different finite elements. `P2` corresponds to a
# continuous piecewise quadratic basis (vector) and `P1` to a continuous
# piecewise linear basis (scalar).


P2 = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,), dtype=default_real_type)
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
V, Q = functionspace(msh, P2), functionspace(msh, P1)

# Boundary conditions for the velocity field are defined:

# +
# No-slip condition on boundaries where x = 0, x = 1, and y = 0
noslip = np.zeros(msh.geometry.dim, dtype=PETSc.ScalarType)  # type: ignore
facets = locate_entities_boundary(msh, 1, noslip_boundary)
bc0 = dirichletbc(noslip, locate_dofs_topological(V, 1, facets), V)

# Driving (lid) velocity condition on top boundary (y = 1)
lid_velocity = Function(V)
lid_velocity.interpolate(lid_velocity_expression)
facets = locate_entities_boundary(msh, 1, lid)
bc1 = dirichletbc(lid_velocity, locate_dofs_topological(V, 1, facets))

# Collect Dirichlet boundary conditions
bcs = [bc0, bc1]
# -

# The bilinear and linear forms for the Stokes equations are defined
# using a a blocked structure:

# +
# Define variational problem
(u, p) = ufl.TrialFunction(V), ufl.TrialFunction(Q)
(v, q) = ufl.TestFunction(V), ufl.TestFunction(Q)
f = Constant(msh, (PETSc.ScalarType(0), PETSc.ScalarType(0)))  # type: ignore

a = form([[inner(grad(u), grad(v)) * dx, inner(p, div(v)) * dx], [inner(div(u), q) * dx, None]])
L = form([inner(f, v) * dx, inner(Constant(msh, PETSc.ScalarType(0)), q) * dx])  # type: ignore
# -

# A block-diagonal preconditioner will be used with the iterative
# solvers for this problem:

a_p11 = form(inner(p, q) * dx)
a_p = [[a[0][0], None], [None, a_p11]]

# ### Nested matrix solver
#
# We assemble the bilinear form into a nested matrix `A`, and call the
# `assemble()` method to communicate shared entries in parallel. Rows
# and columns in `A` that correspond to degrees-of-freedom with
# Dirichlet boundary conditions wil be zeroed by the assembler, and a
# value of 1 will be set on the diagonal for these rows.


def nested_iterative_solver():
    """Solve the Stokes problem using nest matrices and an iterative solver."""

    # Assemble nested matrix operators
    A = fem.petsc.assemble_matrix_nest(a, bcs=bcs)
    A.assemble()

    # Create a nested matrix P to use as the preconditioner. The
    # top-left block of P is shared with the top-left block of A. The
    # bottom-right diagonal entry is assembled from the form a_p11:
    P11 = fem.petsc.assemble_matrix(a_p11, [])
    P = PETSc.Mat().createNest([[A.getNestSubMatrix(0, 0), None], [None, P11]])
    P.assemble()

    A00 = A.getNestSubMatrix(0, 0)
    A00.setOption(PETSc.Mat.Option.SPD, True)

    P00, P11 = P.getNestSubMatrix(0, 0), P.getNestSubMatrix(1, 1)
    P00.setOption(PETSc.Mat.Option.SPD, True)
    P11.setOption(PETSc.Mat.Option.SPD, True)

    # Assemble right-hand side vector
    b = fem.petsc.assemble_vector_nest(L)

    # Modify ('lift') the RHS for Dirichlet boundary conditions
    fem.petsc.apply_lifting_nest(b, a, bcs=bcs)

    # Sum contributions for vector entries that are share across
    # parallel processes
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Set Dirichlet boundary condition values in the RHS vector
    bcs0 = fem.bcs_by_block(extract_function_spaces(L), bcs)
    fem.petsc.set_bc_nest(b, bcs0)

    # The pressure field is determined only up to a constant. We supply
    # a vector that spans the nullspace to the solver, and any component
    # of the solution in this direction will be eliminated during the
    # solution process.
    null_vec = fem.petsc.create_vector_nest(L)

    # Set velocity part to zero and the pressure part to a non-zero
    # constant
    null_vecs = null_vec.getNestSubVecs()
    null_vecs[0].set(0.0), null_vecs[1].set(1.0)

    # Normalize the vector that spans the nullspace, create a nullspace
    # object, and attach it to the matrix
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    assert nsp.test(A)
    A.setNullSpace(nsp)

    # Create a MINRES Krylov solver and a block-diagonal preconditioner
    # using PETSc's additive fieldsplit preconditioner
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A, P)
    ksp.setType("minres")
    ksp.setTolerances(rtol=1e-9)
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    # Define the matrix blocks in the preconditioner with the velocity
    # and pressure matrix index sets
    nested_IS = P.getNestISs()
    ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]))

    # Set the preconditioners for each block. For the top-left
    # Laplace-type operator we use algebraic multigrid. For the
    # lower-right block we use a Jacobi preconditioner. By default, GAMG
    # will infer the correct near-nullspace from the matrix block size.
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")

    # Create finite element {py:class}`Function <dolfinx.fem.Function>`s
    # for the velocity (on the space `V`) and for the pressure (on the
    # space `Q`). The vectors for `u` and `p` are combined to form a
    # nested vector and the system is solved.
    u, p = Function(V), Function(Q)
    x = PETSc.Vec().createNest([la.create_petsc_vector_wrap(u.x), la.create_petsc_vector_wrap(p.x)])
    ksp.solve(b, x)

    # Before writing to file, ghost values are updated using `scatter_forward`.
    u.x.scatter_forward()
    P1 = element(
        "Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,), dtype=default_real_type
    )
    u1 = Function(functionspace(msh, P1))
    u1.interpolate(u)
    u_vtx = io.VTXWriter(MPI.COMM_WORLD, "out_stokes/velocity.bp", [u1], engine="BP5")
    u_vtx.write(0)

    p.x.scatter_forward()
    p_vtx = io.VTXWriter(MPI.COMM_WORLD, "out_stokes/pressure.bp", [p], engine="BP5")
    p_vtx.write(0)

    # Compute norms of the solution vectors
    norm_u = la.norm(u.x)
    norm_p = la.norm(p.x)
    if MPI.COMM_WORLD.rank == 0:
        print(f"(A) Norm of velocity coefficient vector (nested, iterative): {norm_u}")
        print(f"(A) Norm of pressure coefficient vector (nested, iterative): {norm_p}")

    return norm_u, norm_p


# Solve using PETSc MatNest
norm_u_0, norm_p_0 = nested_iterative_solver()
