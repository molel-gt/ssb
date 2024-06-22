#!/usr/bin/env python
# coding: utf-8

# # Solving Full Cell Simulation Using the Discontinuous Galerkin Method with Interior Penalty
# Author: E Leshinka Molel
# 
# In this notebook we set to solve the full cell simulation equations for a solid-state battery. The full-cell is simplified to include three domains
# - negative electrode
# - solid electrolyte (SE) separator
# - positive electrode
# 
# At the interface between negative electrode and SE, and the interface between SE and positive electrode, we have charge transfer reactions. For simplicity, linear kinetics are assumed.
# 
# Phase 1 is the solid active material and phase 2 is the SE.
# 
# Current flowing due to charge transfer reaction is given by linear kinetics expression:
# $$i = \frac{F i_{o}}{RT}(\phi_1 - \phi_2 - U)$$
# 
# The above expression can be written twice, once for -ve electrode and once for +ve electrode.
# 
# In our case, for simplicity, infinite kinetics are assumed at the negative electrode and SE separator.

# ## Setting Up Variational Formulation
# $$\nabla \cdot (-\kappa \nabla u) = f$$
# $$u(x=0,y) = u_0$$
# $$u(x=L_x,y) = u_{L_x}$$
# $$-\kappa \nabla u \cdot \hat{n}|_{y=0,y=L_y} = 0$$
# 
# The total domain is indicated by $\Omega$ and the external boundary of the domain by $\partial \Omega$.
# 
# Because of the internal discontinuity in $u$, we use Discontinuous Lagrange elements. We solve the partial differential equation for each element $K_i$ then sum over all the $N$ elements. This involves multiplying by a test function $v$ and integrating over the element $K_i$.
# 
# $$\sum_i^N \int_{K_i} -\nabla \cdot(\kappa u_i) v \mathrm{dx} = \sum_i^N \int_{K_i} f_i v_i \mathrm{dx}$$
# Integrating the LHS by parts, we obtain:
# $$\sum_i^N \int_{K_i} -\kappa \nabla u_i \cdot \nabla v_i \mathrm{dx} + \sum_i^N \int_{\partial K_i} \kappa \hat{n_i} \cdot \nabla u_i \cdot v_i \mathrm{ds} = \sum_i^N \int_{K_i} f_i v_i \mathrm{dx}$$
# 
# We can split the second term on the LHS to have the parts of the external boundary: $\bigcup \partial \Omega_i = \partial \Omega$ for $i=1,...,N$. Also $\partial \Omega_i \bigcup \partial \hat{K_i} = \partial K_i$.
# 
# Therefore, we have:
# $$\int_{\Omega} -\kappa \nabla u \cdot \nabla v \mathrm{dx} + \int_{\partial \Omega} \kappa \hat{n} \cdot \nabla u \cdot v \mathrm{ds} + \sum_i^N \int_{\partial \hat{K_i}} \kappa \hat{n_i} \cdot \nabla u_i \cdot v_i \mathrm{ds} = \int_{\Omega} f v \mathrm{dx}$$
# 
# The second term on the LHS can be used to set the Neumann boundary conditions where we have those. Below, we expound further on the third term on the LHS.
# 
# We note that for the third term, we have an integral for each side of the integral facet, say between cell $i$ and $j$. We can therefore write this integral as a jump integral (noting that $n_i=-n_j$) over the set of all internal facets $E_j, j=0,\dots,N_{internal}$
# 
# \begin{align}
# \sum_i^N \int_{\partial \hat{K_i}} \kappa \hat{n_i} \cdot \nabla u_i \cdot v_i \mathrm{ds} = -\sum_j^{N_{internal}} \int_{E_j} \kappa \hat{n_j} \cdot [\nabla u_j \cdot v_j] \mathrm{ds}
# \end{align}
# 
# We can further expand the RHS of above using the relation $[a \cdot b] = \langle a\rangle \cdot [b] + [a] \cdot \langle b \rangle$. Where $\langle \text{value} \rangle$ denotes average of the value across the shared internal boundary. Therefore, we can rewrite our expression above into:
# 
# \begin{align}
# \sum_i^N \int_{\partial \hat{K_i}} \kappa \hat{n_i} \cdot \nabla u_i \cdot v_i \mathrm{ds} = -\sum_j^{N_{internal}} \int_{E_j} \kappa \hat{n_j} \cdot [\nabla u_j] \cdot \langle v_j\rangle \mathrm{ds} -\sum_j^{N_{internal}} \int_{E_j} \kappa \hat{n_j} \cdot \langle \nabla u_j\rangle \cdot [v_j] \mathrm{ds}
# \end{align}
# 
# Because we want our solution to be conservative, we enforce that the jump of the gradient in normal direction is zero by removing the term involving $[\nabla u]$ from the RHS of the above expression.
# 
# To maintain symmetry when $u$ and $v$ are switched, we add a term $-\sum_j^{N_{internal}} \int_{E_j} \kappa \hat{n_j} \cdot \langle \nabla v_j\rangle \cdot [u_j] \mathrm{ds}$ to the RHS. We also add another term to the RHS for coercivity: $\int_{E_j}\frac{\gamma}{\langle h \rangle}[u][v]~\mathrm{d}s$ where $h$ is the diameter of the circumscribed circle.
# 
# We use Nitsche's method to impose Dirichlet and Neumann boundary conditions on the exterior boundary $\partial \Omega$.

import os

import dolfinx
import gmsh
# import matplotlib.pyplot as plt
# import meshio
import numpy as np
import ufl
import warnings

from basix.ufl import element
from dolfinx import cpp, default_scalar_type, fem, graph, io, mesh, nls, plot
from dolfinx.fem import petsc
from dolfinx.graph import partitioner_parmetis
from dolfinx.io import gmshio, VTXWriter
from dolfinx.nls import petsc as petsc_nls
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 dot, div, dx, ds, dS, grad, inner, grad, avg, jump)

import commons, utils

warnings.simplefilter('ignore')

dtype = PETSc.ScalarType


class NewtonSolver:
    max_iterations: int
    bcs: list[fem.DirichletBC]
    A: PETSc.Mat
    b: PETSc.Vec
    J: fem.Form
    b: fem.Form
    dx: PETSc.Vec

    def __init__(
        self,
        F: list[fem.form],
        J: list[list[fem.form]],
        w: list[fem.Function],
        bcs: list[fem.DirichletBC] | None = None,
        max_iterations: int = 5,
        petsc_options: dict[str, str | float | int | None] = None,
        problem_prefix="newton",
    ):
        self.max_iterations = max_iterations
        self.bcs = [] if bcs is None else bcs
        self.b = fem.petsc.create_vector_block(F)
        self.F = F
        self.J = J
        self.A = fem.petsc.create_matrix_block(J)
        self.dx = self.A.createVecLeft()
        self.w = w
        self.x = fem.petsc.create_vector_block(F)

        # Set PETSc options
        opts = PETSc.Options()
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v

        # Define KSP solver
        self._solver = PETSc.KSP().create(self.b.getComm().tompi4py())
        self._solver.setOperators(self.A)
        self._solver.setFromOptions()

        # Set matrix and vector PETSc options
        self.A.setFromOptions()
        self.b.setFromOptions()

    def solve(self, tol=1e-6, beta=1.0):
        i = 0

        while i < self.max_iterations:
            dolfinx.cpp.la.petsc.scatter_local_vectors(
                self.x,
                [si.x.petsc_vec.array_r for si in self.w],
                [
                    (
                        si.function_space.dofmap.index_map,
                        si.function_space.dofmap.index_map_bs,
                    )
                    for si in self.w
                ],
            )
            self.x.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

            # Assemble F(u_{i-1}) - J(u_D - u_{i-1}) and set du|_bc= u_D - u_{i-1}
            with self.b.localForm() as b_local:
                b_local.set(0.0)
            fem.petsc.assemble_vector_block(
                self.b, self.F, self.J, bcs=self.bcs, x0=self.x, scale=-1.0
            )
            self.b.ghostUpdate(
                PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD
            )


            # Assemble Jacobian
            self.A.zeroEntries()
            fem.petsc.assemble_matrix_block(self.A, self.J, bcs=self.bcs)
            self.A.assemble()

            self._solver.solve(self.b, self.dx)
            # self._solver.view()
            assert (
                self._solver.getConvergedReason() > 0
            ), "Linear solver did not converge"
            offset_start = 0
            for s in self.w:
                num_sub_dofs = (
                    s.function_space.dofmap.index_map.size_local
                    * s.function_space.dofmap.index_map_bs
                )
                s.x.petsc_vec.array_w[:num_sub_dofs] -= (
                    beta * self.dx.array_r[offset_start : offset_start + num_sub_dofs]
                )
                s.x.petsc_vec.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
                )
                offset_start += num_sub_dofs
            # Compute norm of update

            correction_norm = self.dx.norm(0)
            print(f"Iteration {i}: Correction norm {correction_norm}")
            if correction_norm < tol:
                break
            i += 1

    def __del__(self):
        self.A.destroy()
        self.b.destroy()
        self.dx.destroy()
        self._solver.destroy()
        self.x.destroy()


def arctanh(y):
    return 0.5 * ufl.ln((1 + y) / (1 - y))


def ocv(c, cmax=30000):
    xi = 2 * (c - 0.5 * cmax) / cmax
    return 3.25 - 0.5 * arctanh(xi)


Wa = 1e1
encoding = io.XDMFFile.Encoding.HDF5
comm = MPI.COMM_WORLD
micron = 1e-6
markers = commons.Markers()
LX = 150 * micron
LY = 40 * micron

# workdir = "output/subdomains_dg/150-40-0/20-55-20/1e-07/"
workdir = "output/tertiary_current/150-40-0/20-55-20/1.0e-06/"
utils.make_dir_if_missing(workdir)
output_meshfile = os.path.join(workdir, 'mesh.msh')
potential_resultsfile = os.path.join(workdir, "potential.bp")
concentration_resultsfile = os.path.join(workdir, "concentration.bp")
current_resultsfile = os.path.join(workdir, "current.bp")

R = 8.3145
T = 298
faraday_const = 96485
kappa_elec = 0.1
kappa_pos_am = 0.2
i0 = kappa_elec * R * T / faraday_const / Wa / LX

partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
domain, ct, ft = gmshio.read_from_msh(output_meshfile, comm, partitioner=partitioner)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(tdim, fdim)

ft_imap = domain.topology.index_map(fdim)
num_facets = ft_imap.size_local + ft_imap.num_ghosts
indices = np.arange(0, num_facets)
values = np.zeros(indices.shape, dtype=np.intc)  # all facets are tagged with zero

values[ft.indices] = ft.values
ft = mesh.meshtags(domain, fdim, indices, values)

# create submesh
submesh, entity_map, vertex_map, geom_map = mesh.create_submesh(
    domain, tdim, ct.find(markers.positive_am)
)
# transfer tags from parent to submesh
tdim = domain.topology.dim
fdim = tdim - 1
c_to_f = domain.topology.connectivity(tdim, fdim)
f_map = domain.topology.index_map(fdim)
all_facets = f_map.size_local + f_map.num_ghosts
all_values = np.zeros(all_facets, dtype=np.int32)
all_values[ft.indices] = ft.values

submesh.topology.create_entities(fdim)
subf_map = submesh.topology.index_map(fdim)
submesh.topology.create_connectivity(tdim, fdim)
submesh.topology.create_connectivity(tdim, tdim)
submesh.topology.create_connectivity(fdim, fdim)
c_to_f_sub = submesh.topology.connectivity(tdim, fdim)
num_sub_facets = subf_map.size_local + subf_map.num_ghosts
sub_values = np.empty(num_sub_facets, dtype=np.int32)
for i, entity in enumerate(entity_map):
    parent_facets = c_to_f.links(entity)
    child_facets = c_to_f_sub.links(i)
    for child, parent in zip(child_facets, parent_facets):
        sub_values[child] = all_values[parent]
submesh_ft = mesh.meshtags(submesh, submesh.topology.dim - 1, np.arange(
    num_sub_facets, dtype=np.int32), sub_values)
submesh.topology.create_connectivity(submesh.topology.dim - 1, submesh.topology.dim)

# entity_maps = {submesh: entity_map, domain: ct.indices}
mesh_to_submesh = np.full(len(ct.indices), -1)
mesh_to_submesh[entity_map] = np.arange(len(entity_map))
entity_maps = {submesh: mesh_to_submesh, domain: ct.indices}

# integration measures
dx = ufl.Measure("dx", domain=domain, subdomain_data=ct)
dx_r = ufl.Measure("dx", domain=submesh)
ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
ds_r = ufl.Measure("ds", domain=submesh, subdomain_data=submesh_ft)

f_to_c = domain.topology.connectivity(fdim, tdim)
c_to_f = domain.topology.connectivity(tdim, fdim)
charge_xfer_facets = ft.find(markers.electrolyte_v_positive_am)

other_internal_facets = ft.find(0)
int_facet_domain = []
for f in charge_xfer_facets:
    if f >= ft_imap.size_local or len(f_to_c.links(f)) != 2:
        continue
    c_0, c_1 = f_to_c.links(f)[0], f_to_c.links(f)[1]
    subdomain_0, subdomain_1 = ct.values[[c_0, c_1]]
    local_f_0 = np.where(c_to_f.links(c_0) == f)[0][0]
    local_f_1 = np.where(c_to_f.links(c_1) == f)[0][0]
    if subdomain_0 > subdomain_1:
        int_facet_domain.append(c_0)
        int_facet_domain.append(local_f_0)
        int_facet_domain.append(c_1)
        int_facet_domain.append(local_f_1)
    else:
        int_facet_domain.append(c_1)
        int_facet_domain.append(local_f_1)
        int_facet_domain.append(c_0)
        int_facet_domain.append(local_f_0)

other_internal_facet_domains = []
for f in other_internal_facets:
    if f >= ft_imap.size_local or len(f_to_c.links(f)) != 2:
        continue
    c_0, c_1 = f_to_c.links(f)[0], f_to_c.links(f)[1]
    subdomain_0, subdomain_1 = ct.values[[c_0, c_1]]
    local_f_0 = np.where(c_to_f.links(c_0) == f)[0][0]
    local_f_1 = np.where(c_to_f.links(c_1) == f)[0][0]
    other_internal_facet_domains.append(c_0)
    other_internal_facet_domains.append(local_f_0)
    other_internal_facet_domains.append(c_1)
    other_internal_facet_domains.append(local_f_1)
int_facet_domains = [(markers.electrolyte_v_positive_am, int_facet_domain)]#, (0, other_internal_facet_domains)]

dS = ufl.Measure("dS", domain=domain, subdomain_data=int_facet_domains)


# concentrations for nmc622
nmc_622_density = 2.597e3
area_electrolyte = fem.assemble_scalar(fem.form(1.0 * dx(markers.electrolyte)))
area_positive_am = fem.assemble_scalar(fem.form(1.0 * dx(markers.positive_am)))
print(f"Electrolyte Area: {area_electrolyte:.2e}")
print(f"Positive AM Area: {area_positive_am:.2e}")
# assume 1 meter
positive_capacity = utils.nmc_capacity(nmc_622_density, area_positive_am * 1)
print(f"Positive Capacity: {positive_capacity * 1000:.3f} [mA.h]")
c_li = utils.lithium_concentration_nmc(nmc_622_density)
print(f"Lithium concentration in NMC622: {c_li} [mol/m3]")


# ### Function Spaces

V = fem.functionspace(domain, ("DG", 1))
V_submesh = fem.functionspace(submesh, ("CG", 1))
W = fem.functionspace(domain, ("DG", 1, (3,)))
Q = fem.functionspace(domain, ("DG", 0))
u = fem.Function(V, name='potential')
c = fem.Function(V_submesh, name='concentration')
c0 = fem.Function(V_submesh, name='concentration')
c0.interpolate(lambda x: c_li + x[0] - x[0])
c.interpolate(c0)
q = ufl.TestFunction(V_submesh)
# u_cg = fem.Function(V_CG, name='potential', dtype=np.float64)
v = ufl.TestFunction(V)
n = ufl.FacetNormal(domain)
nc = ufl.FacetNormal(submesh)
x = ufl.SpatialCoordinate(domain)

h = ufl.CellDiameter(domain)
h_avg = avg(h)


# constants

f = fem.Constant(domain, dtype(0))
fc = fem.Constant(submesh, dtype(0))
g = fem.Constant(domain, dtype(0))
gc = fem.Constant(submesh, dtype(0))
voltage = 1
u_left = fem.Function(V)
with u_left.vector.localForm() as u0_loc:
    u0_loc.set(0)
u_right = fem.Function(V)
with u_right.vector.localForm() as u1_loc:
    u1_loc.set(voltage)


# #### $\kappa$ varying in each domain

kappa = fem.Function(Q)
cells_electrolyte = ct.find(markers.electrolyte)
cells_pos_am = ct.find(markers.positive_am)
kappa.x.array[cells_electrolyte] = np.full_like(cells_electrolyte, kappa_elec, dtype=dtype)
kappa.x.array[cells_pos_am] = np.full_like(cells_pos_am, kappa_pos_am, dtype=dtype)
D = 1e-15

# ### variational formulation

alpha = 100
gamma = 100
Wa_n = 1e-3
Wa_p = 1e1
i0_n = kappa_elec * R * T / (Wa_n * faraday_const * LX)
i0_p = kappa_elec * R * T / (Wa_p * faraday_const * LX)

# potential problem
i_loc = -inner((kappa * grad(u))('+'), n("+"))
u_jump = 2 * ufl.ln(0.5 * i_loc/i0_p + ufl.sqrt((0.5 * i_loc/i0_p)**2 + 1)) * (R * T / faraday_const)

u_ocv = ocv(c("+"))
F0 = kappa * inner(grad(u), grad(v)) * dx - f * v * dx - kappa * inner(grad(u), n) * v * ds

# Add DG/IP terms
F0 += - avg(kappa) * inner(jump(u, n), avg(grad(v))) * dS#(0)
F0 += - inner(avg(kappa * grad(u)), jump(v, n)) * dS#(0)
F0 += alpha / h_avg * avg(kappa) * inner(jump(v, n), jump(u, n)) * dS#(0)

# Internal boundary
F0 += + avg(kappa) * dot(avg(grad(v)), (u_jump + u_ocv) * n('+')) * dS(markers.electrolyte_v_positive_am)
F0 += -alpha / h_avg * avg(kappa) * dot(jump(v, n), (u_jump + u_ocv) * n('+')) * dS(markers.electrolyte_v_positive_am)

# # Symmetry
F0 += - avg(kappa) * inner(jump(u, n), avg(grad(v))) * dS(markers.electrolyte_v_positive_am)

# # Coercivity
F0 += alpha / h_avg * avg(kappa) * inner(jump(u, n), jump(v, n)) * dS(markers.electrolyte_v_positive_am)

# Nitsche Dirichlet BC terms on left and right boundaries
F0 += - kappa * (u - u_left) * inner(n, grad(v)) * ds(markers.left)
F0 += -gamma / h * (u - u_left) * v * ds(markers.left)
F0 += - kappa * (u - u_right) * inner(n, grad(v)) * ds(markers.right) 
F0 += -gamma / h * (u - u_right) * v * ds(markers.right)

# Nitsche Neumann BC terms on insulated boundary
F0 += -g * v * ds(markers.insulated_electrolyte) + gamma * h * g * inner(grad(v), n) * ds(markers.insulated_electrolyte)
F0 += - gamma * h * inner(inner(grad(u), n), inner(grad(v), n)) * ds(markers.insulated_electrolyte)
F0 += -g * v * ds(markers.insulated_positive_am) + gamma * h * g * inner(grad(v), n) * ds(markers.insulated_positive_am)
F0 += - gamma * h * inner(inner(grad(u), n), inner(grad(v), n)) * ds(markers.insulated_positive_am)

# kinetics boundary - neumann
# F += - gamma * h * inner(inner(kappa * grad(u), n), inner(grad(v), n)) * ds(markers.left)
# F -= - gamma * h * 2 * i0_n * ufl.sinh(0.5 * faraday_const / R / T * (V_left - u - 0)) * inner(grad(v), n) * ds(markers.left)

# concentration problem

dt = 1e-3
F1 = inner(c - c0, q) * dx_r + dt * inner(D * grad(c), grad(q)) * dx_r
F1 -= dt * (inner(fc, q) * dx_r + inner(gc, q) * (ds_r(markers.insulated_positive_am) + ds_r(markers.right)) - inner(1/faraday_const * inner(-kappa * grad(u), n), q) * ds_r(markers.electrolyte_v_positive_am))


# solve tertiary current distribution

TIME = 1000 * dt
t = 0
c_vtx = VTXWriter(comm, concentration_resultsfile, [c], engine="BP4")
c_vtx.write(0.0)

while t < TIME:
    t += dt
    jac00 = ufl.derivative(F0, u)
    jac01 = ufl.derivative(F0, c)
    jac10 = ufl.derivative(F1, u)
    jac11 = ufl.derivative(F1, c)
    
    J00 = fem.form(jac00, entity_maps=entity_maps)
    J01 = fem.form(jac01, entity_maps=entity_maps)
    J10 = fem.form(jac10, entity_maps=entity_maps)
    J11 = fem.form(jac11, entity_maps=entity_maps)
    
    J = [[J00, J01], [J10, J11]]
    F = [
        fem.form(F0, entity_maps=entity_maps),
        fem.form(F1, entity_maps=entity_maps),
        ]
    solver = NewtonSolver(
        F,
        J,
        [u, c],
        bcs=[],
        max_iterations=2,
        petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "superlu_dist",
        },
        )
    solver.solve(1e-5, beta=1)
    c0.x.array[:] = c.x.array[:]
    c_vtx.write(t)
c_vtx.close()


# ### variational problem solution

area_left = fem.assemble_scalar(fem.form(1.0 * ds(markers.left)))
area_right = fem.assemble_scalar(fem.form(1.0 * ds(markers.right)))
u_avg_right = fem.assemble_scalar(fem.form(u * ds(markers.right))) / area_right
u_avg_left = fem.assemble_scalar(fem.form(u * ds(markers.left))) / area_left
u_stdev_right = np.sqrt(fem.assemble_scalar(fem.form((u - u_avg_right) ** 2 * ds(markers.right))) / area_right)
u_stdev_left = np.sqrt(fem.assemble_scalar(fem.form((u - u_avg_left) ** 2 * ds(markers.left))) / area_left)
print(f"Left - avg potential  : {u_avg_left:.3e}, stdev potential  : {u_stdev_left:.3e}")
print(f"Right - avg potential : {u_avg_right:.3e}, stdev potential  : {u_stdev_right:.3e}")
