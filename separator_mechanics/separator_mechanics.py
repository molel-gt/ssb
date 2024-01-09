# -*- coding: utf-8 -*-
from basix.ufl import element
from ufl import (Coefficient, FunctionSpace, Identity, Mesh, TestFunction,
                 TrialFunction, derivative, det, diff, dx, grad, ln, tr,
                 variable)


# Function spaces
e = element("Lagrange", "tetrahedron", 1, shape=(3,))
mesh = Mesh(e)
V = FunctionSpace(mesh, e)

# Trial and test functions
du = TrialFunction(V)     # Incremental displacement
v = TestFunction(V)      # Test function

# Functions
u = Coefficient(V)        # Displacement from previous iteration
# B = Coefficient(element)        # Body force per unit volume
# T = Coefficient(element)        # Traction force on the boundary

# Now, we can define the kinematic quantities involved in the model::

# Kinematics
d = len(u)
I = Identity(d)
F = variable(I + grad(u))
C = F.T * F

# Invariants of deformation tensors
Ic = tr(C)
J = det(F)

# Elasticity parameters
E = 10.0
ν = 0.3
μ = E / (2 * (1 + ν))
λ = E * ν / ((1 + ν) * (1 - 2 * ν))

# Stored strain energy density (compressible neo-Hookean model)
Ψ = (μ / 2) * (Ic - 3) - μ * ln(J) + (λ / 2) * (ln(J))**2

# Total potential energy
Pi = Ψ * dx  # - inner(B, u) * dx - inner(T, u) * ds

# First variation of Pi (directional derivative about u in the direction of v)
F_form = derivative(Pi, u, v)

# Compute Jacobian of F
J_form = derivative(F_form, u, du)

# Compute Cauchy stress
sigma = (1 / J) * diff(Ψ, F) * F.T

forms = [F_form, J_form]
elements = [e]
expressions = [(sigma, [[0.25, 0.25, 0.25]])]
