import dolfinx
import numpy as np
import ufl
import time
import warnings

import scifem

from basix.ufl import element
from dolfinx import cpp, default_scalar_type, fem, graph, io, mesh, nls, plot
from dolfinx.fem import petsc

from dolfinx.io import gmshio, VTXWriter
from dolfinx.nls import petsc as petsc_nls
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 dot, div, dx, ds, dS, grad, inner, grad, avg, jump)


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
        self._solver.setOperators(self.A, self.A)
        self._solver.setFromOptions()

        # Set matrix and vector PETSc options
        self.A.setFromOptions()
        self.b.setFromOptions()
        # self._solver.setMonitor(lambda _, it, residual: PETSc.Sys.Print(it, residual))
        self._solver.setTolerances(rtol=1e-7)
        self._solver.view()

    def solve(self, tol=1e-7, beta=1.0):
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
                self.b, self.F, self.J, bcs=self.bcs, x0=self.x, alpha=-1.0
            )
            self.b.ghostUpdate(
                PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD
            )


            # Assemble Jacobian
            self.A.zeroEntries()
            fem.petsc.assemble_matrix_block(self.A, self.J, bcs=self.bcs)
            self.A.assemble()

            self._solver.solve(self.b, self.dx)

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
            PETSc.Sys.Print(f"Iteration {i}: Correction norm {correction_norm}")
            if correction_norm < tol:
                break
            if np.isnan(self.dx.norm(0)):
                break
            i += 1

    def __del__(self):
        self.A.destroy()
        self.b.destroy()
        self.dx.destroy()
        self._solver.destroy()
        self.x.destroy()


class NonlinearPDE_SNESProblem:
    def __init__(self, F, J, soln_vars, bcs, P=None, entity_maps=None):
        self.L = F
        self.a = J
        self.a_precon = P
        self.bcs = bcs
        self.soln_vars = soln_vars

    def F_mono(self, snes, x, F):
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting, assemble_vector, set_bc

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with x.localForm() as _x:
            self.soln_vars.x.array[:] = _x.array_r
        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, self.bcs, x, -1.0)

    def J_mono(self, snes, x, J, P):
        from dolfinx.fem.petsc import assemble_matrix

        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            assemble_matrix(P, self.a_precon, bcs=self.bcs, diagonal=1.0)
            P.assemble()

    def F_block(self, snes, x, F):
        from petsc4py import PETSc

        from dolfinx.fem.petsc import assemble_vector_block

        assert x.getType() != "nest"
        assert F.getType() != "nest"
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with F.localForm() as f_local:
            f_local.set(0.0)

        offset = 0
        x_array = x.getArray(readonly=True)
        for var in self.soln_vars:
            size_local = var.x.petsc_vec.getLocalSize()
            var.x.petsc_vec.array[:] = x_array[offset : offset + size_local]
            var.x.petsc_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )
            offset += size_local

        assemble_vector_block(F, self.L, self.a, bcs=self.bcs, x0=x, alpha=-1.0)

    def J_block(self, snes, x, J, P):
        from dolfinx.fem.petsc import assemble_matrix_block

        assert x.getType() != "nest" and J.getType() != "nest" and P.getType() != "nest"
        J.zeroEntries()
        assemble_matrix_block(J, self.a, bcs=self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            assemble_matrix_block(P, self.a_precon, bcs=self.bcs, diagonal=1.0)
            P.assemble()

    def F_nest(self, snes, x, F):
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting, assemble_vector, set_bc

        assert x.getType() == "nest" and F.getType() == "nest"
        # Update solution
        x = x.getNestSubVecs()
        for x_sub, var_sub in zip(x, self.soln_vars):
            x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            with x_sub.localForm() as _x:
                var_sub.x.array[:] = _x.array_r

        # Assemble
        bcs1 = fem.bcs_by_block(fem.extract_function_spaces(self.a, 1), self.bcs)
        for L, F_sub, a in zip(self.L, F.getNestSubVecs(), self.a):
            with F_sub.localForm() as F_sub_local:
                F_sub_local.set(0.0)
            assemble_vector(F_sub, L)
            apply_lifting(F_sub, a, bcs=bcs1, x0=x, alpha=-1.0)
            F_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set bc value in RHS
        bcs0 = fem.bcs_by_block(fem.extract_function_spaces(self.L), self.bcs)
        for F_sub, bc, x_sub in zip(F.getNestSubVecs(), bcs0, x):
            set_bc(F_sub, bc, x_sub, -1.0)

        # Must assemble F here in the case of nest matrices
        F.assemble()

    def J_nest(self, snes, x, J, P):
        from dolfinx.fem.petsc import assemble_matrix_nest

        assert J.getType() == "nest" and P.getType() == "nest"
        J.zeroEntries()
        assemble_matrix_nest(J, self.a, bcs=self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            assemble_matrix_nest(P, self.a_precon, bcs=self.bcs, diagonal=1.0)
            P.assemble()



class ShuntCurrentsSolver:
    def __init__(self, comm, N_s=100, d_p=0.01, V_cell=1.0, kappa=4.0, H_p=0.03,
                 A_m=0.006, L_p=0.02, a=100, i0=10, a_a=0.5, a_c=0.5):
        self._N_s = N_s
        self._d_p = d_p
        self._V_cell = V_cell
        self._kappa = kappa
        self._H_p = H_p
        self._A_m = A_m
        self._L_p = L_p
        self._a = a
        self._i0 = i0
        self._a_a = a_a
        self._a_c = a_c
        self._comm = comm
        self._domain = None
        self._u_lin = None
        self._u_bv = None
        self._eta_s = None
        self._eta_s_0 = None
        self._v = None
        self._V = None
        self._faraday_constant = 96485
        self._R = 8.314
        self._T = 298

    @property
    def N_s(self):
        return self._N_s

    @property
    def d_p(self):
        return self._d_p

    @property
    def V_cell(self):
        return self._V_cell

    @property
    def kappa(self):
        return self._kappa

    @property
    def H_p(self):
        return self._H_p

    @property
    def A_m(self):
        return self._A_m

    @property
    def L_p(self):
        return self._L_p

    @property
    def a(self):
        return self._a

    @property
    def i0(self):
        return self._i0

    @property
    def a_a(self):
        return self._a_a

    @property
    def a_c(self):
        return self._a_c

    @property
    def L(self):
        return 0.5 * self.N_s * self.d_p

    @property
    def R_p(self):
        return self.L_p / self.kappa

    @property
    def dx(self):
        return self._dx

    @property
    def bcs(self):
        return self._bcs

    @property
    def x(self):
        return self._x

    @property
    def n(self):
        return self._n

    @property
    def ds(self):
        return self._ds

    @property
    def faraday_constant(self):
        return self._faraday_constant

    @property
    def R(self):
        return self._R

    @property
    def T(self):
        return self._T

    @property
    def comm(self):
        return self._comm

    @property
    def domain(self):
        return self._domain

    @property
    def u(self):
        return self._u

    @property
    def V(self):
        return self._V

    @property
    def v(self):
        return self._v

    @property
    def eta_s_0(self):
        return self._eta_s_0

    @property
    def n_its(self):
        return self._n_its

    @property
    def converged_bv(self):
        return self._converged_bv

    def _A(self):
        return ((2 * self.a * self.i0 * self.kappa * self.R * self.T) / (self.a_a * self.a_c * self.faraday_constant)) ** 0.5

    def _B(self, eta_s):
        return self.a_c * ufl.exp(self.a_a * self.faraday_constant * eta_s / self. R / self.T) +\
            self.a_a * ufl.exp(-self.a_c * self.faraday_constant * eta_s / self. R / self.T) - self.a_a - self.a_c

    def i_p(self, eta_s):
        return self._A() * self._B(eta_s) ** 0.5

    def di_p_deta_s(self, eta_s):
        return 0.5 * self._A() * self._B(eta_s) ** -0.5 * (
                    self.a_a * self.a_c * self.faraday_constant / self.R / self.T) * (
                        ufl.exp(self.a_a * self.faraday_constant * eta_s / self.R * self.T)-\
                        ufl.exp(-self.a_c * self.faraday_constant * eta_s / self.R / self.T)
                        )

    def phi_linear(self):
        return (-self.V_cell/(self.d_p * self.lmbda))/(ufl.exp(self.lmbda * self.L) + ufl.exp(-self.lmbda * self.L)) * (ufl.exp(self.lmbda * self.x[0]) - ufl.exp(-self.lmbda * self.x[0])) + self.V_cell * self.x[0] / self.d_p

    def phi_bv(self):
        return self.phi_linear() + self.i_p(self.eta_s_0)/self.di_p_deta_s(self.eta_s_0) - self.eta_s_0

    def setup(self):
        self._domain = mesh.create_interval(self.comm, 1000, [0, self.L])
        tdim = self.domain.topology.dim
        fdim = tdim - 1
        ft_imap = self.domain.topology.index_map(fdim)
        num_facets = ft_imap.size_local + ft_imap.num_ghosts
        indices = np.arange(0, num_facets)
        values = np.zeros(indices.shape, dtype=np.intc)
        left_marker = 1
        right_marker = 2

        values[0] = left_marker
        values[-1] = right_marker
        ft = mesh.meshtags(self.domain, fdim, indices, values)

        self._x = ufl.SpatialCoordinate(self.domain)
        self._n = ufl.FacetNormal(self.domain)
        self._V = fem.functionspace(self.domain, ("CG", 3))

        self._u, self._v = fem.Function(self.V), ufl.TestFunction(self.V)
        self._eta_s_0 = fem.Function(self.V)
        self._eta_s_0.interpolate(lambda x: x[0] - x[0] + 1e-8)

        self._dx = ufl.Measure('dx', domain=self.domain)
        self._ds = ufl.Measure('ds', domain=self.domain, subdomain_data=ft)

        u_left = fem.Function(self.V)
        u_left.x.array[:] = 0
        self.domain.topology.create_connectivity(fdim, tdim)
        left_bc = fem.dirichletbc(
            u_left, fem.locate_dofs_topological(self.V, fdim, ft.find(left_marker))
        )

        self._bcs = [left_bc]

    @property
    def omega(self):
        return np.sqrt(self.a * self.i0 * (self.a_a + self.a_c) * self.faraday_constant / self.kappa / self.R / self.T)

    @property
    def lmbda(self):
        return np.sqrt(self.H_p / self.A_m * (1/(self.L_p + 1/self.omega)))

    def lambda_squared(self, eta_s):
        return self.H_p / (self.kappa * self.A_m) * self.di_p_deta_s(eta_s) / (self.di_p_deta_s(eta_s) * self.R_p + 1)

    def f(self, y, eta_s):
        return self.lambda_squared(eta_s) * (self.V_cell / self.d_p * y + self.i_p(eta_s)/self.di_p_deta_s(eta_s) - self.eta_s_0)

    def solve_bv(self, tol=1e-8, max_its=10):
        self._n_its = 0
        error = tol + 1
        x_fun = fem.Function(self.V)
        x_fun.interpolate(lambda x: x[0])

        while error > tol and self.n_its < max_its:
            F0 = -inner(self.kappa * grad(self.u), grad(self.v)) * self.dx
            F0 += - self.H_p * self.di_p_deta_s(self.eta_s_0)/(self.kappa * self.A_m * (1 + self.R_p * self.di_p_deta_s(self.eta_s_0))) * self.u * self.v * self.dx
            F0 += + self.H_p * self.di_p_deta_s(self.eta_s_0)/(self.kappa * self.A_m * (1 + self.R_p * self.di_p_deta_s(self.eta_s_0))) * (self.V_cell / self.d_p * self.x[0] + self.i_p(self.eta_s_0)/self.di_p_deta_s(self.eta_s_0) - self.eta_s_0) * self.v * self.dx
            F = [fem.form(F0)]
            j00 = fem.form(ufl.derivative(F0, self.u))
            J = [[j00]]
            opts = {
                        'ksp_type': 'fgmres',
                        'pc_type': 'hypre',
                        }

            solver = scifem.NewtonSolver(F, J, [self.u], bcs=self.bcs, petsc_options=opts)

            t0 = time.time()
            solver.solve()
            t1 = time.time()
            error = np.sqrt(fem.assemble_scalar(fem.form(((self.x[0] - 2 * self.u/self.N_s - self.eta_s_0)) ** 2 * self.dx)))
            self._eta_s_0.x.array[:] = x_fun.x.array - 2 * self.u.x.array / self.N_s
            PETSc.Sys.Print(f"Iteration: {self.n_its}, Error: {error:.2e}, Solve time: {t1 - t0:.3f}s")
            self._n_its += 1
        self._converged_bv = (error <= tol and self.n_its <= max_its)
        if self._converged_bv:
            PETSc.Sys.Print(f"Converged in {self.n_its}!")
        else:
            PETSc.Sys.Print(f"Failed to converge!")
