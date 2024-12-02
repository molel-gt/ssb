import dolfinx
import gmsh
import numpy as np
import ufl
import warnings

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


class BlockNewtonSolver:
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
        iset=[],
        petsc_options: dict[str, str | float | int | None] = None,
        problem_prefix="newton",
    ):
        self.max_iterations = max_iterations
        self.bcs = [] if bcs is None else bcs
        self.b = fem.petsc.create_vector_block(F)
        self.b_u = fem.petsc.create_vector_block([F[0], F[1]])
        self.b_u0 = fem.petsc.create_vector(F[0])
        self.b_u1 = fem.petsc.create_vector(F[1])
        self.b_c = fem.petsc.create_vector(F[2])
        self.F = F
        self.J = J
        self.A = fem.petsc.create_matrix_block(J)
        self.dx = self.A.createVecLeft()
        self.w = w
        self.x = fem.petsc.create_vector_block(F)
        self.comm = self.b.getComm().tompi4py()
        self.iset = iset
        self.iset_u = self.iset[0].sum(self.iset[1])

        self.A_u = fem.petsc.create_matrix_block([[J[0][0], J[0][1]], [J[1][0], J[1][1]]])
        self.A_c = fem.petsc.create_matrix(J[2][2])
        self.A01_2 = fem.petsc.assemble_matrix_block([[self.J[0][2]],[self.J[1][2]]])
        self.A02 = fem.petsc.assemble_matrix(self.J[0][2])
        self.A12 = fem.petsc.assemble_matrix(self.J[1][2])
        self.A21 = fem.petsc.assemble_matrix(self.J[2][1])

        # Set PETSc options
        opts = PETSc.Options()
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v

        ######################### Define KSP solver ############################
        # potential block solver
        # self.ksp_u = PETSc.KSP().create(self.comm)
        # self.ksp_u.setType(PETSc.KSP.Type.PREONLY)
        # self.ksp_u.setTolerances(rtol=1.0e-7, max_it=100)
        # self.ksp_u.setMonitor(lambda _, it, residual: print(it, residual))
        # self.ksp_u.setErrorIfNotConverged(True)
        # self.ksp_u.setOperators(self.A_u)
        # self.ksp_u.getPC().setType(PETSc.PC.Type.HYPRE)
        # self.ksp_u.getPC().setType("fieldsplit")
        # self.ksp_u.getPC().setFieldSplitIS(("u0", self.iset[0]), ("u1", self.iset[1]))
        # self.ksp_u0, self.ksp_u1 = self.ksp_u.getPC().getFieldSplitSubKSP()
        # self.ksp_u.getPC().setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        # self.ksp_u.getPC().setFieldSplitSchurPreType(PETSc.PC.SchurPreType.FULL)
        # self.ksp_u.getPC().setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)

        # self.ksp_u0.setType(PETSc.KSP.Type.FGMRES)
        # self.ksp_u0.getPC().setType(PETSc.PC.Type.HYPRE)
        # self.ksp_u1.setType(PETSc.KSP.Type.FGMRES)
        # self.ksp_u1.getPC().setType(PETSc.PC.Type.HYPRE)
        # self.ksp_u0.setErrorIfNotConverged(True)
        # self.ksp_u1.setErrorIfNotConverged(True)
        # self.ksp_u.setFromOptions()
        # self.ksp_u0.setFromOptions()
        # self.ksp_u1.setFromOptions()

        # concentration solver
        # self.ksp_c = PETSc.KSP().create(self.comm)
        # self.ksp_c.setType(PETSc.KSP.Type.GMRES)
        # self.ksp_c.setOperators(self.A_c)
        # self.ksp_c.setTolerances(rtol=1.0e-7, max_it=100)
        # self.ksp_c.setErrorIfNotConverged(True)
        # self.ksp_c.setMonitor(lambda _, it, residual: print(it, residual))
        # self.ksp_c.getPC().setType(PETSc.PC.Type.HYPRE)
        # self.ksp_c.getPC().setHYPREType("boomeramg")
        # self.ksp_c.setFromOptions()

        self._solver = PETSc.KSP().create(self.comm)
        self._solver.setOperators(self.A, self.A)
        self._solver.setFromOptions()

        # Set matrix and vector PETSc options
        self.A.setFromOptions()
        self.b.setFromOptions()

    def solve_u(self, x_u0, x_u, x_c0):
        b_u0 = x_u.duplicate()
        b_u = b_u0.duplicate()
        b_u.zeroEntries()
        b_c = x_c0.duplicate()

        self.b.getSubVector(self.iset[2], subvec=b_c)
        self.b.getSubVector(self.iset[-1], subvec=b_u0)

        self.A01_2.mult(x_c0, b_u)
        b_u.scale(-1.0)
        b_u += b_u0

        self.ksp_u.solve(b_u, x_u)
        x_u.scale(0.001)
        x_u += x_u0

        self.b.restoreSubVector(self.iset[-1], subvec=b_u0)
        self.b.restoreSubVector(self.iset[2], subvec=b_c)

    def solve_c(self, x_u0, x_c0, x_c):
        x_u0 = self.A02.createVecLeft()
        x_u1 = self.A12.createVecLeft()
        b_u0 = x_u0.duplicate()
        b_u1 = x_u1.duplicate()
        self.x.getSubVector(self.iset[1], subvec=x_u1)
        self.x.getSubVector(self.iset[0], subvec=x_u0)

        self.b.getSubVector(self.iset[2], subvec=self.b_c)
        self.b.getSubVector(self.iset[1], subvec=b_u1)
        self.b.getSubVector(self.iset[0], subvec=b_u0)
        u1 = x_u1.duplicate()
        b_c0 = x_c.duplicate()
        x_u_sol = x_u0.duplicate()

        self.A21.mult(u1, x_c0)
        self.b_c += -b_c0
        self.ksp_c.solve(self.b_c, x_c)
        x_c.scale(0.01)
        x_c += x_c0
        self.x.restoreSubVector(self.iset[0], subvec=x_u0)
        self.x.restoreSubVector(self.iset[1], subvec=x_u1)

        self.b.restoreSubVector(self.iset[0], subvec=b_u0)
        self.b.restoreSubVector(self.iset[1], subvec=b_u1)
        self.b.restoreSubVector(self.iset[2], subvec=self.b_c)

    def linear_solve(self, tol=1e-3, max_its=2):
        error = 1.0
        x_u0 = self.A_u.createVecLeft()
        x_c0 = self.A_c.createVecLeft()
        x_u_prev = x_u0.duplicate()
        x_c_prev = x_c0.duplicate()
        e_vec_c = x_c_prev.duplicate()
        its = 0
        x_u = x_u0.duplicate()
        x_u.zeroEntries()
        x_c = x_c0.duplicate()
        x_c.zeroEntries()
        self.dx.getSubVector(self.iset[2], subvec=x_c)
        self.dx.getSubVector(self.iset[-1], subvec=x_u)
        x_u_prev = x_u.duplicate()
        x_c_prev = x_c.duplicate()
        # PETSc.Sys.Print(f"Init residue: {self.dx.norm()}")
        while error > tol and its < max_its:
            PETSc.Sys.Print(x_c_prev.norm())
            self.solve_c(x_u_prev, x_c_prev, x_c)
            PETSc.Sys.Print(x_c.norm())
        
            # self.x.getSubVector(self.iset[2], subvec=self.x_c)
            evec_c = x_c - x_c_prev
            error_c = evec_c.norm(0)
            x_c_prev = x_c.duplicate()
            self.solve_u(x_u_prev, x_u, x_c)
            evec_u = x_u_prev - x_u
            error_u = evec_u.norm(0)
            x_u_prev = x_u.duplicate()
            # PETSc.Sys.Print(f"Inner Iteration: {its}, c: {error_c}, u: {error_u}")
            error = max(error_c, error_u)
            its += 1
            PETSc.Sys.Print(f"It: {its}, r: {self.dx.norm(0)}, error: {error}")
        self.dx.restoreSubVector(self.iset[-1], subvec=x_u)
        self.dx.restoreSubVector(self.iset[2], subvec=x_c)

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
                self.b, self.F, self.J, bcs=self.bcs, x0=self.x, alpha=-1.0
            )

            self.b.ghostUpdate(
                PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD
            )

            # # Assemble Jacobian
            self.A.zeroEntries()
            fem.petsc.assemble_matrix_block(self.A, self.J, bcs=self.bcs)
            self.A.assemble()

            # self.A_c.zeroEntries()
            # fem.petsc.assemble_matrix(self.A_c, self.J[2][2], bcs=self.bcs)
            # self.A_c.assemble()

            # self.A01_2.zeroEntries()
            # fem.petsc.assemble_matrix_block(self.A01_2, [[self.J[0][2]], [self.J[1][2]]], bcs=self.bcs)
            # self.A01_2.assemble()

            # # self.A02.zeroEntries()
            # fem.petsc.assemble_matrix(self.A02, self.J[0][2], bcs=self.bcs)
            # self.A02.assemble()

            # # self.A12.zeroEntries()
            # fem.petsc.assemble_matrix(self.A12, self.J[1][2], bcs=self.bcs)
            # self.A12.assemble()

            # # self.A21.zeroEntries()
            # fem.petsc.assemble_matrix(self.A21, self.J[2][1], bcs=self.bcs)
            # self.A21.assemble()

            # self.A_u.zeroEntries()
            # fem.petsc.assemble_matrix_block(self.A_u, [[self.J[0][0], self.J[0][1]], [self.J[1][0], self.J[1][1]]], bcs=self.bcs)
            # self.A_u.assemble()
            # # solve linear system
            # self.linear_solve()

            self._solver.solve(self.b, self.dx)
            # self._solver.view()
            # self._solver.setMonitor(lambda _, it, residual: print(it, residual))
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
            # # Compute norm of update

            correction_norm = self.dx.norm(0)
            PETSc.Sys.Print(f"Outer Iteration {i}: Correction norm {correction_norm}")
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
        # self.ksp_u.destroy()
        # self.ksp_u0.destroy()
        # self.ksp_u1.destroy()
        # self.ksp_u.destroy()
        self.x.destroy()
