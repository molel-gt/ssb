from fenics import *
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt


def solver(f, u_D, Nx, Ny, Nz, degree=1):
    """
    Solve -Laplace(u) = f on [0,1] x [0,1] with 2*Nx*Ny Lagrange
    elements of specified degree and u = u_D (Expresssion) on
    the boundary.
    """
    # Create mesh and define function space
    mesh = Mesh()
    fp = XDMFFile("mesh.xdmf")
    fp.read(mesh)
    V = FunctionSpace(mesh, 'P', degree)
    tol = 1E-14

    def boundary_D(x, on_boundary):
        return on_boundary and (near(x[0], 0, tol) or near(x[0], 1, tol))

    bc = DirichletBC(V, u_D, boundary_D)

    # Define variational problem
    u = TrialFunction(V)
    g = Expression('x[1]*(1 - x[1])*x[2]*(1 - x[2])', degree=2)
    v = TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    L = f*v*dx - g*v*ds

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)
    return u


def run_solver():
    """
    Run solver to compute and post-process solution
    """
    # Set up problem parameters and call solver
    u_D = Expression('x[0]', degree=2)

    f = Constant('0')
    u = solver(f, u_D, 10, 10, 10, 1)
    # Plot solution and mesh
    plot(u)
    # plot(u.function_space().mesh())
    # Save solution to file in VTK format
    vtkfile = File('poisson_solver/solution.pvd')
    vtkfile << u
    plt.show()


if __name__ == '__main__':
    run_solver()
    # interactive()
