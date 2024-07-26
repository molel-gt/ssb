import gmsh
import os
import numpy as np
import tqdm.autonotebook

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, io, plot

from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector, 
                               create_vector, create_matrix, set_bc)
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio)

from ufl import (FacetNormal, finiteelement, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym)
from ufl import sqrt, sin


L = 200
H = 100
c_x = c_y = 0.25
r = 0.05
gdim = 2
fdim = gdim - 1

dtype = PETSc.ScalarType

# partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
domain, ct, ft = io.gmshio.read_from_msh("mesh.msh", MPI.COMM_WORLD, rank=0, gdim=gdim)
inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5


t = 0.0
T = 2.0
num_steps = 2000
dt = T / num_steps
dt = fem.Constant(domain, dtype(dt))
mu = fem.Constant(domain, dtype(1.0e-4))
rho = fem.Constant(domain, dtype(1.0))
nu = fem.Constant(domain, dtype(mu/rho))
f = fem.Constant(domain, dtype((0, 0)))

resolution = dtype(r)
Cs = dtype(0.035)  # Constante de Smagorinsky


V = fem.functionspace(domain, ("CG", 2, (2, )))
Q = fem.functionspace(domain, ("CG", 1))


class InletVelocity():
    def __init__(self, t):
        self.t = t
        self.tol = 1e-4
        self.alpha_atm = 0.3

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]),dtype=PETSc.ScalarType)
        values[0] = 4 * 1.5 * sin(self.t * np.pi/8) * x[1] * (H - x[1])/H**2
        # values[0] = 4.0 * ((x[1] + self.tol) / H)**self.alpha_atm
        return values

# Inlet
u_inlet = fem.Function(V)
inlet_velocity = InletVelocity(t)
u_inlet.interpolate(inlet_velocity)
bcu_inflow = fem.dirichletbc(u_inlet, fem.locate_dofs_topological(V, fdim, ft.find(inlet_marker)))
# Walls
u_nonslip = np.array((0,) * domain.geometry.dim, dtype=dtype)
u0 = fem.Function(V)
with u0.vector.localForm() as u0_loc:
    u0_loc.set(0)
bcu_walls = fem.dirichletbc(u_nonslip, fem.locate_dofs_topological(V, fdim, ft.find(wall_marker)), V)
# Obstacle
bcu_obstacle = fem.dirichletbc(u_nonslip, fem.locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V)
bcu = [bcu_inflow, bcu_obstacle, bcu_walls]

# Outlet
bcp_outlet = fem.dirichletbc(dtype(0.0), fem.locate_dofs_topological(Q, fdim, ft.find(outlet_marker)), Q)
bcp = [bcp_outlet]


def Sij(w):
    """
        # Strain-rate tensor
    """
    return sym(nabla_grad(w))


def nu_T(w):
    """
        Turbulence viscosity
    """
    return resolution**2 * Cs**2 * sqrt(2.0 * inner(Sij(w), Sij(w)))


# =============================================================================
#                        Variational Formulation
# =============================================================================

u_tent = TrialFunction(V)
v = TestFunction(V)
u_ = fem.Function(V)
u_.name = "u"
u_n = fem.Function(V)

p = TrialFunction(Q)
q = TestFunction(Q)
p_ = fem.Function(Q)


# Incompressible
F1 = dot(u_tent - u_n, v) * dx + \
    dt * inner(dot(u_n, nabla_grad(u_n)), v) * dx + \
    dt * (nu + nu_T(u_n)) * inner(grad(u_tent), grad(v)) * dx #+ \
    # dt * inner(Sij(u_tent), Sij(u_tent)) * inner(grad(u_tent), grad(v))
    # 2.0 * dt * nu_T(u_n) * inner(Sij(u_tent), Sij(v)) * dx

a1 = fem.form(lhs(F1))
L1 = fem.form(rhs(F1))
A1 = create_matrix(a1)
b1 = create_vector(L1)

# Pressure
a2 = fem.form(dot(grad(p), grad(q)) * dx)
L2 = fem.form(-rho/dt * dot(div(u_), q) * dx)
A2 = assemble_matrix(a2, bcs=bcp)
A2.assemble()
b2 = create_vector(L2)

# Velocity
a3 = fem.form(dot(u_tent, v) * dx)
L3 = fem.form(dot(u_, v) * dx - dt/rho * dot(grad(p_), v) * dx)
A3 = assemble_matrix(a3)
A3.assemble()
b3 = create_vector(L3)


# =============================================================================
#                                   SOLVERS
# =============================================================================

# Solver for step 1
solver1 = PETSc.KSP().create(domain.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.JACOBI)

# Solver for step 2
solver2 = PETSc.KSP().create(domain.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.MINRES)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Solver for step 3
solver3 = PETSc.KSP().create(domain.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)


# # # Archivos de salida
vtx_u = io.VTXWriter(domain.comm, "u-les.bp", [u_])
vtx_p = io.VTXWriter(domain.comm, "p-lens.bp", [p_])
vtx_u.write(t)
vtx_p.write(t)

progress = tqdm.tqdm(desc="Solving PDE", total=num_steps)
for i in range(num_steps):
    progress.update(1)
    t += dt.value
    inlet_velocity.t = t
    u_inlet.interpolate(inlet_velocity)
    
    # PASO 1
    A1.zeroEntries()
    assemble_matrix(A1, a1, bcs=bcu)
    A1.assemble()
    with b1.localForm() as loc:
        loc.set(0)
    
    assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [bcu])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    solver1.solve(b1, u_.vector)
    u_.x.scatter_forward()
    
    # PASO 2
    with b2.localForm() as loc:
        loc.set(0)
    
    assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [bcp])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    solver2.solve(b2, p_.vector)
    p_.x.scatter_forward()
    
    # PASO 3
    with b3.localForm() as loc:
        loc.set(0)
    
    assemble_vector(b3, L3)
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(b3, u_n.vector)
    u_n.x.scatter_forward()
    
    # write to file
    vtx_u.write(t)
    vtx_p.write(t)

vtx_u.close()
vtx_p.close()
