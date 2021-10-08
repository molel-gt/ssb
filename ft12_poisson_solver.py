"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  -Laplace(u) = f    in the unit square
            u = u_D  on the boundary

  u = 1 + x^2 + 2y^2 = u_D
  f = -6

This is an extended version of the demo program poisson.py which
encapsulates the solver as a Python function.
"""

from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt


TOL = 1e-4


def solver(f, u_D, Nx, Ny, Nz, degree=1):
    """
    Solve -Laplace(u) = f on [0,1] x [0,1] with 2*Nx*Ny Lagrange
    elements of specified degree and u = u_D (Expresssion) on
    the boundary.
    """

    # Create mesh and define function space
    mesh = UnitSquareMesh(Nx, Ny)
    V = FunctionSpace(mesh, 'P', degree)

    # Define boundary condition
    def boundary(x, on_boundary):
        return on_boundary and (near(x[0], 0, TOL) or near(x[0], 1, TOL))

    bc = DirichletBC(V, u_D, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    g = Expression('x[1]*(1 - x[1])', degree=1)
    L = f*v*dx - g*v*ds

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    return u


def run_solver():
    "Run solver to compute and post-process solution"

    # Set up problem parameters and call solver
    u_D = Expression('x[0]', degree=2)
    f = Constant(0)
    u = solver(f, u_D, 10, 10, 10, 1)

    # Plot solution and mesh
    plot(u)
    plot(u.function_space().mesh())

    # Save solution to file in VTK format
    vtkfile = File('poisson_solver/solution.pvd')
    vtkfile << u


if __name__ == '__main__':
    run_solver()
    plt.show()
