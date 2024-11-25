import sys, time
import numpy as np
from petsc4py import PETSc


class BlockPreconditioner:
    def __init__(self, comm, iset, precond_fields, solver_params={}):
        self._comm = comm
        self._iset = iset
        self._precond_fields = precond_fields
        self.schur_block_scaling = [{'type': 'diag', 'val': -1.0}, {'type': 'diag', 'val': -1.0}, {'type': 'diag', 'val': -1.0}]

    @property
    def comm(self):
        return self._comm

    @property
    def iset(self):
        return self._iset

    @property
    def size(self):
        return len(self.precond_fields)

    @property
    def precond_fields(self):
        return self._precond_fields

    def create(self, pc):
        _, self.P = pc.getOperators()
        opts = PETSc.Options()
        operator_mats = self.init_mat_vec(pc)

        self.ksp_fields, self.ksp_py_solver = [], [None]*self.size

        for n in range(self.size):
            self.ksp_fields.append( PETSc.KSP().create(self.comm) )

        for n in range(self.size):
            if self.precond_fields[n]['prec'] == 'amg':
                try: solvetype = self.precond_fields[n]['solve']
                except: solvetype = "preonly"
                self.ksp_fields[n].setType(solvetype)
                # GMRES or FGMES for inner solve
                if solvetype == 'gmres' or solvetype == 'fgmres':
                    try: maxiter = self.precond_fields[n]['maxiter']
                    except: maxiter = 1000
                    try: rtol = self.precond_fields[n]['rtol']
                    except: rtol = 1e-7
                    try: atol = self.precond_fields[n]['atol']
                    except: atol = 1e-50
                    self.ksp_fields[n].setTolerances(rtol=rtol, atol=atol, divtol=None, max_it=maxiter)
                try: amgtype = self.precond_fields[n]['amgtype']
                except: amgtype = "hypre"
                self.ksp_fields[n].getPC().setType(amgtype)
                if amgtype=="hypre":
                    self.ksp_fields[n].getPC().setHYPREType("boomeramg")
                # add PETSc options
                if 'petsc_options' in self.precond_fields[n].keys():
                    opt_dict = self.precond_fields[n]['petsc_options']
                    for o in opt_dict:
                        opts.setValue(o, opt_dict[o])
                    self.ksp_fields[n].setFromOptions() # solver options
                    self.ksp_fields[n].getPC().setFromOptions() # preconditioner options
                    for key in opts.getAll(): opts.delValue(key) # clear options - opts.clear() doesn't seem to work?!
                # print to view some settings...
                #print(self.ksp_fields[n].getPC().view())
            elif self.precond_fields[n]['prec'] == 'direct':
                self.ksp_fields[n].setType("preonly")
                self.ksp_fields[n].getPC().setType("lu")
                self.ksp_fields[n].getPC().setFactorSolverType("mumps")
            else:
                self.ksp_fields[n].setType(self.precond_fields[n]['solve'])
                self.ksp_fields[n].getPC().setType(self.precond_fields[n]['prec'])
                # raise ValueError("Unknown preconditioner type")

            self.ksp_fields[n].setOperators(operator_mats[n])

    def init_mat_vec(self, pc):
        self.A  = self.P.createSubMatrix(self.iset[0],self.iset[0])
        self.A.assemble()
        self.Bt = self.P.createSubMatrix(self.iset[0],self.iset[1])
        self.Bt.assemble()
        self.Dt = self.P.createSubMatrix(self.iset[0],self.iset[2])
        self.Dt.assemble()
        self.B  = self.P.createSubMatrix(self.iset[1],self.iset[0])
        self.B.assemble()
        self.C  = self.P.createSubMatrix(self.iset[1],self.iset[1])
        self.C.assemble()
        self.Et = self.P.createSubMatrix(self.iset[1],self.iset[2])
        self.Et.assemble()
        self.D  = self.P.createSubMatrix(self.iset[2],self.iset[0])
        self.D.assemble()
        self.E  = self.P.createSubMatrix(self.iset[2],self.iset[1])
        self.E.assemble()
        self.R  = self.P.createSubMatrix(self.iset[2],self.iset[2])
        self.R.assemble()

        # the matrix to later insert the diagonal
        self.Adinv = PETSc.Mat().createAIJ(self.A.getSizes(), bsize=None, nnz=(1,1), csr=None, comm=self.comm)
        self.Adinv.setUp()
        self.Adinv.assemble()
        # set 1's to get correct allocation pattern
        self.Adinv.shift(1.)

        if self.schur_block_scaling[0]['type']=='diag':
            self.adinv_vec = self.A.getDiagonal()
        elif self.schur_block_scaling[0]['type']=='rowsum':
            self.adinv_vec = self.A.getRowSum()
        elif self.schur_block_scaling[0]['type']=='none':
            self.adinv_vec = self.A.createVecLeft()
            self.adinv_vec.set(1.0)
        else:
            raise ValueError("Unknown schur_block_scaling option!")

        self.Smod = self.C.copy(structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

        if self.schur_block_scaling[1]['type']=='diag':
            self.smoddinv_vec = self.Smod.getDiagonal()
        elif self.schur_block_scaling[1]['type']=='rowsum':
            self.smoddinv_vec = self.Smod.getRowSum()
        elif self.schur_block_scaling[1]['type']=='none':
            self.smoddinv_vec = self.Smod.createVecLeft()
            self.smoddinv_vec.set(1.0)
        else:
            raise ValueError("Unknown schur_block_scaling option!")

        # the matrix to later insert the diagonal
        self.Smoddinv = PETSc.Mat().createAIJ(self.C.getSizes(), bsize=None, nnz=(1,1), csr=None, comm=self.comm)
        self.Smoddinv.setUp()
        self.Smoddinv.assemble()
        # set 1's to get correct allocation pattern
        self.Smoddinv.shift(1.)

        self.Tmod = self.Et.copy(structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        self.Umod = self.E.copy(structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        self.Wmod = self.R.copy(structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

        self.Adinv_Bt = self.Adinv.matMult(self.Bt)
        self.D_Adinv_Bt = self.D.matMult(self.Adinv_Bt)

        self.B_Adinv_Bt = self.B.matMult(self.Adinv_Bt)

        self.Adinv_Dt = self.Adinv.matMult(self.Dt)
        self.B_Adinv_Dt = self.B.matMult(self.Adinv_Dt)

        self.D_Adinv_Dt = self.D.matMult(self.Adinv_Dt)

        # need to set Smod and Tmod here to get the data structures right
        self.Smod.axpy(-1., self.B_Adinv_Bt)
        self.Umod.axpy(-1., self.D_Adinv_Bt)
        self.Tmod.axpy(-1., self.B_Adinv_Dt)

        self.Smoddinv_Tmod = self.Smoddinv.matMult(self.Tmod)

        self.Umod_Smoddinv_Tmod = self.Umod.matMult(self.Smoddinv_Tmod)

        self.By1 = self.B.createVecLeft()
        self.Dy1 = self.D.createVecLeft()
        self.Umody2 = self.E.createVecLeft()
        self.Tmody3 = self.Et.createVecLeft()
        self.Bty2 = self.Bt.createVecLeft()
        self.Dty3 = self.Dt.createVecLeft()

        self.x1, self.x2, self.x3 = self.A.createVecLeft(), self.Smod.createVecLeft(), self.Wmod.createVecLeft()
        self.y1, self.y2, self.y3 = self.A.createVecLeft(), self.Smod.createVecLeft(), self.Wmod.createVecLeft()
        self.z1, self.z2, self.z3 = self.A.createVecLeft(), self.Smod.createVecLeft(), self.Wmod.createVecLeft()

        # do we need these???
        self.A.setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, True)
        self.Smod.setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, True)
        self.Wmod.setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, True)

        return [self.A, self.Smod, self.Wmod]

    def setUp(self, pc):
        self.P.createSubMatrix(self.iset[0],self.iset[0], submat=self.A)
        self.P.createSubMatrix(self.iset[0],self.iset[1], submat=self.Bt)
        self.P.createSubMatrix(self.iset[0],self.iset[2], submat=self.Dt)
        self.P.createSubMatrix(self.iset[1],self.iset[0], submat=self.B)
        self.P.createSubMatrix(self.iset[1],self.iset[1], submat=self.C)
        self.P.createSubMatrix(self.iset[1],self.iset[2], submat=self.Et)
        self.P.createSubMatrix(self.iset[2],self.iset[0], submat=self.D)
        self.P.createSubMatrix(self.iset[2],self.iset[1], submat=self.E)
        self.P.createSubMatrix(self.iset[2],self.iset[2], submat=self.R)

        if self.schur_block_scaling[0]['type']=='diag':
            self.A.getDiagonal(result=self.adinv_vec)
            self.adinv_vec.reciprocal()
        elif self.schur_block_scaling[0]['type']=='rowsum':
            self.A.getRowSum(result=self.adinv_vec)
            self.adinv_vec.abs()
            self.adinv_vec.reciprocal()
        elif self.schur_block_scaling[0]['type']=='none':
            self.adinv_vec.set(1.0)
        else:
            raise ValueError("Unknown schur_block_scaling option!")

        self.adinv_vec.scale(self.schur_block_scaling[0]['val'])

        # form diag(A)^{-1}
        self.Adinv.setDiagonal(self.adinv_vec, addv=PETSc.InsertMode.INSERT)

        self.Adinv.matMult(self.Bt, result=self.Adinv_Bt)      # diag(A)^{-1} Bt
        self.B.matMult(self.Adinv_Bt, result=self.B_Adinv_Bt)  # B diag(A)^{-1} Bt

        # --- modified Schur complement Smod = C - B diag(A)^{-1} Bt
        # compute self.Smod = self.C - B_Adinv_Bt
        self.C.copy(result=self.Smod)
        self.Smod.axpy(-1., self.B_Adinv_Bt)

        # --- Umod = E - D diag(A)^{-1} Bt
        # --- Tmod = Et - B diag(A)^{-1} Dt

        self.Adinv.matMult(self.Dt, result=self.Adinv_Dt)      # diag(A)^{-1} Dt
        self.B.matMult(self.Adinv_Dt, result=self.B_Adinv_Dt)  # B diag(A)^{-1} Dt
        self.D.matMult(self.Adinv_Bt, result=self.D_Adinv_Bt)  # D diag(A)^{-1} Bt

        # compute self.Umod = self.E - D_Adinv_Bt
        self.E.copy(result=self.Umod)
        self.Umod.axpy(-1., self.D_Adinv_Bt)

        # compute self.Tmod = self.Et - B_Adinv_Dt
        self.Et.copy(result=self.Tmod)
        self.Tmod.axpy(-1., self.B_Adinv_Dt)

        # --- Wmod = R - D diag(A)^{-1} Dt - Umod diag(Smod)^{-1} Tmod

        if self.schur_block_scaling[1]['type']=='diag':
            self.Smod.getDiagonal(result=self.smoddinv_vec)
            self.smoddinv_vec.reciprocal()
        elif self.schur_block_scaling[1]['type']=='rowsum':
            self.Smod.getRowSum(result=self.smoddinv_vec)
            self.smoddinv_vec.abs()
            self.smoddinv_vec.reciprocal()
        elif self.schur_block_scaling[1]['type']=='none':
            self.smoddinv_vec.set(1.0)
        else:
            raise ValueError("Unknown schur_block_scaling option!")

        self.smoddinv_vec.scale(self.schur_block_scaling[1]['val'])

        # form diag(Smod)^{-1}
        self.Smoddinv.setDiagonal(self.smoddinv_vec, addv=PETSc.InsertMode.INSERT)

        self.Smoddinv.matMult(self.Tmod, result=self.Smoddinv_Tmod)                        # diag(Smod)^{-1} Tmod

        self.Umod.matMult(self.Smoddinv_Tmod, result=self.Umod_Smoddinv_Tmod)              # Umod diag(Smod)^{-1} Tmod

        self.D.matMult(self.Adinv_Dt, result=self.D_Adinv_Dt)                              # D diag(A)^{-1} Dt

        # compute self.Wmod = self.R - D_Adinv_Dt - Umod_Smoddinv_Tmod
        self.R.copy(result=self.Wmod)
        self.Wmod.axpy(-1., self.D_Adinv_Dt)
        self.Wmod.axpy(-1., self.Umod_Smoddinv_Tmod)

        # operator values have changed - do we need to re-set them?
        self.ksp_fields[0].setOperators(self.A)
        self.ksp_fields[1].setOperators(self.Smod)
        self.ksp_fields[2].setOperators(self.Wmod)

    def apply(self, pc, x, y):
        # get subvectors (references!)
        x.getSubVector(self.iset[0], subvec=self.x1)
        x.getSubVector(self.iset[1], subvec=self.x2)
        x.getSubVector(self.iset[2], subvec=self.x3)

        tss = time.time()

        # 1) solve A * y_1 = x_1
        self.ksp_fields[0].solve(self.x1, self.y1)

        self.B.mult(self.y1, self.By1)

        # compute z2 = x2 - self.By1
        self.z2.axpby(1., 0., self.x2)
        self.z2.axpy(-1., self.By1)

        # 2) solve Smod * y_2 = z_2
        self.ksp_fields[1].solve(self.z2, self.y2)

        self.D.mult(self.y1, self.Dy1)
        self.Umod.mult(self.y2, self.Umody2)

        # compute z3 = x3 - self.Dy1 - self.Umody2
        self.z3.axpby(1., 0., self.x3)
        self.z3.axpy(-1., self.Dy1)
        self.z3.axpy(-1., self.Umody2)

        # 3) solve Wmod * y_3 = z_3
        self.ksp_fields[2].solve(self.z3, self.y3)

        self.Tmod.mult(self.y3, self.Tmody3)

        # compute z2 = x2 - self.By1 - self.Tmody3
        self.z2.axpy(-1., self.Tmody3)

        # 4) solve Smod * y_2 = z_2
        self.ksp_fields[1].solve(self.z2, self.y2)

        self.Bt.mult(self.y2, self.Bty2)
        self.Dt.mult(self.y3, self.Dty3)

        # compute z1 = x1 - self.Bty2 - self.Dty3
        self.z1.axpby(1., 0., self.x1)
        self.z1.axpy(-1., self.Bty2)
        self.z1.axpy(-1., self.Dty3)

        # 5) solve A * y_1 = z_1
        self.ksp_fields[0].solve(self.z1, self.y1)

        # restore/clean up
        x.restoreSubVector(self.iset[0], subvec=self.x1)
        x.restoreSubVector(self.iset[1], subvec=self.x2)
        x.restoreSubVector(self.iset[2], subvec=self.x3)

        # set into y vector
        y.setNestSubVecs([self.y1, self.y2, self.y3])
        # y.setValues(self.iset[0].getIndices(), self.y1.array)
        # y.setValues(self.iset[1], self.y2.array)
        # y.setValues(self.iset[2], self.y3.array)

        y.assemble()

    def destroy(self, pc):
        pc.destroy()


class BGSPreconditioner:
    def __init__(self, comm, iset, precond_fields, solver_params={}):
        self._comm = comm
        self._iset = iset
        self._precond_fields = precond_fields
        self.schur_block_scaling = [{'type': 'diag', 'val': 1.0}, {'type': 'diag', 'val': 1.0}, {'type': 'diag', 'val': 1.0}]

    @property
    def comm(self):
        return self._comm

    @property
    def iset(self):
        return self._iset

    @property
    def size(self):
        return len(self.precond_fields)

    @property
    def precond_fields(self):
        return self._precond_fields

    def create(self, pc):
        _, self.P = pc.getOperators()
        opts = PETSc.Options()
        operator_mats = self.init_mat_vec(pc)

        self.ksp_fields, self.ksp_py_solver = [], [None]*self.size

        for n in range(self.size):
            self.ksp_fields.append( PETSc.KSP().create(self.comm) )

        for n in range(self.size):
            if self.precond_fields[n]['prec'] == 'amg':
                try: solvetype = self.precond_fields[n]['solve']
                except: solvetype = "preonly"
                self.ksp_fields[n].setType(solvetype)
                # GMRES or FGMES for inner solve
                if solvetype == 'gmres' or solvetype == 'fgmres':
                    try: maxiter = self.precond_fields[n]['maxiter']
                    except: maxiter = 1000
                    try: rtol = self.precond_fields[n]['rtol']
                    except: rtol = 1e-8
                    try: atol = self.precond_fields[n]['atol']
                    except: atol = 1e-50
                    self.ksp_fields[n].setTolerances(rtol=rtol, atol=atol, divtol=None, max_it=maxiter)
                try: amgtype = self.precond_fields[n]['amgtype']
                except: amgtype = "hypre"
                self.ksp_fields[n].getPC().setType(amgtype)
                if amgtype=="hypre":
                    self.ksp_fields[n].getPC().setHYPREType("boomeramg")
                # add PETSc options
                if 'petsc_options' in self.precond_fields[n].keys():
                    opt_dict = self.precond_fields[n]['petsc_options']
                    for o in opt_dict:
                        opts.setValue(o, opt_dict[o])
                    self.ksp_fields[n].setFromOptions() # solver options
                    self.ksp_fields[n].getPC().setFromOptions() # preconditioner options
                    for key in opts.getAll(): opts.delValue(key) # clear options - opts.clear() doesn't seem to work?!
                # print to view some settings...
                #print(self.ksp_fields[n].getPC().view())
            elif self.precond_fields[n]['prec'] == 'direct':
                self.ksp_fields[n].setType("preonly")
                self.ksp_fields[n].getPC().setType("lu")
                self.ksp_fields[n].getPC().setFactorSolverType("mumps")
            else:
                self.ksp_fields[n].setType(self.precond_fields[n]['solve'])
                self.ksp_fields[n].getPC().setType(self.precond_fields[n]['prec'])
                # raise ValueError("Unknown preconditioner type")

            self.ksp_fields[n].setOperators(operator_mats[n])

    def init_mat_vec(self, pc):

        self.A  = self.P.createSubMatrix(self.iset[0],self.iset[0])
        self.B  = self.P.createSubMatrix(self.iset[1],self.iset[0])
        self.C  = self.P.createSubMatrix(self.iset[1],self.iset[1])
        self.D  = self.P.createSubMatrix(self.iset[2],self.iset[0])
        self.E  = self.P.createSubMatrix(self.iset[2],self.iset[1])
        self.R  = self.P.createSubMatrix(self.iset[2],self.iset[2])

        self.By1 = self.B.createVecLeft()
        self.Dy1 = self.D.createVecLeft()
        self.Ey2 = self.E.createVecLeft()

        self.x1, self.x2, self.x3 = self.A.createVecLeft(), self.C.createVecLeft(), self.R.createVecLeft()
        self.y1, self.y2, self.y3 = self.A.createVecLeft(), self.C.createVecLeft(), self.R.createVecLeft()
        self.z2, self.z3 = self.C.createVecLeft(), self.R.createVecLeft()

        # do we need these???
        self.A.setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, True)
        self.C.setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, True)
        self.R.setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, True)

        return [self.A, self.C, self.R]

    def setUp(self, pc):

        ts = time.time()

        self.P.createSubMatrix(self.iset[0],self.iset[0], submat=self.A)
        self.P.createSubMatrix(self.iset[1],self.iset[0], submat=self.B)
        self.P.createSubMatrix(self.iset[1],self.iset[1], submat=self.C)
        self.P.createSubMatrix(self.iset[2],self.iset[0], submat=self.D)
        self.P.createSubMatrix(self.iset[2],self.iset[1], submat=self.E)
        self.P.createSubMatrix(self.iset[2],self.iset[2], submat=self.R)

        # operator values have changed - do we need to re-set them?
        self.ksp_fields[0].setOperators(self.A)
        self.ksp_fields[1].setOperators(self.C)
        self.ksp_fields[2].setOperators(self.R)

    # computes y = P^{-1} x
    def apply(self, pc, x, y):

        # get subvectors
        x.getSubVector(self.iset[0], subvec=self.x1)
        x.getSubVector(self.iset[1], subvec=self.x2)
        x.getSubVector(self.iset[2], subvec=self.x3)

        # 1) solve A * y_1 = x_1
        self.ksp_fields[0].solve(self.x1, self.y1)

        self.B.mult(self.y1, self.By1)

        # compute z2 = x2 - self.By1
        self.z2.axpby(1., 0., self.x2)
        self.z2.axpy(-1., self.By1)

        # 2) solve C * y_2 = z_2
        self.ksp_fields[1].solve(self.z2, self.y2)

        self.D.mult(self.y1, self.Dy1)
        self.E.mult(self.y2, self.Ey2)

        # compute z3 = x3 - self.Dy1 - self.Ey2
        self.z3.axpby(1., 0., self.x3)
        self.z3.axpy(-1., self.Dy1)
        self.z3.axpy(-1., self.Ey2)

        # 3) solve R * y_3 = z_3
        self.ksp_fields[2].solve(self.z3, self.y3)

        # restore/clean up
        x.restoreSubVector(self.iset[0], subvec=self.x1)
        x.restoreSubVector(self.iset[1], subvec=self.x2)
        x.restoreSubVector(self.iset[2], subvec=self.x3)

        # set into y vector
        y.setValues(self.iset[0], self.y1.array)
        y.setValues(self.iset[1], self.y2.array)
        y.setValues(self.iset[2], self.y3.array)

        y.assemble()


    def destroy(self, pc):
        pc.destroy()


class BGSSIMPLEPreconditioner:
    def __init__(self, comm, iset, precond_fields, solver_params={}):
        self._comm = comm
        self._iset = iset
        self._precond_fields = precond_fields
        self.schur_block_scaling = [{'type': 'diag', 'val': 1.0}, {'type': 'diag', 'val': 1.0}, {'type': 'diag', 'val': 1.0}]

    @property
    def comm(self):
        return self._comm

    @property
    def iset(self):
        return self._iset

    @property
    def size(self):
        return len(self.precond_fields)

    @property
    def precond_fields(self):
        return self._precond_fields

    def create(self, pc):
        _, self.P = pc.getOperators()
        opts = PETSc.Options()
        operator_mats = self.init_mat_vec(pc)

        self.ksp_fields, self.ksp_py_solver = [], [None]*self.size

        for n in range(self.size):
            self.ksp_fields.append( PETSc.KSP().create(self.comm) )

        for n in range(self.size):
            if self.precond_fields[n]['prec'] == 'amg':
                try: solvetype = self.precond_fields[n]['solve']
                except: solvetype = "preonly"
                self.ksp_fields[n].setType(solvetype)
                # GMRES or FGMES for inner solve
                if solvetype == 'gmres' or solvetype == 'fgmres':
                    try: maxiter = self.precond_fields[n]['maxiter']
                    except: maxiter = 1000
                    try: rtol = self.precond_fields[n]['rtol']
                    except: rtol = 1e-5
                    try: atol = self.precond_fields[n]['atol']
                    except: atol = 1e-50
                    self.ksp_fields[n].setTolerances(rtol=rtol, atol=atol, divtol=None, max_it=maxiter)
                try: amgtype = self.precond_fields[n]['amgtype']
                except: amgtype = "hypre"
                self.ksp_fields[n].getPC().setType(amgtype)
                if amgtype=="hypre":
                    self.ksp_fields[n].getPC().setHYPREType("boomeramg")
                # add PETSc options
                if 'petsc_options' in self.precond_fields[n].keys():
                    opt_dict = self.precond_fields[n]['petsc_options']
                    for o in opt_dict:
                        opts.setValue(o, opt_dict[o])
                    self.ksp_fields[n].setFromOptions() # solver options
                    self.ksp_fields[n].getPC().setFromOptions() # preconditioner options
                    for key in opts.getAll(): opts.delValue(key) # clear options - opts.clear() doesn't seem to work?!
                # print to view some settings...
                #print(self.ksp_fields[n].getPC().view())
            elif self.precond_fields[n]['prec'] == 'direct':
                self.ksp_fields[n].setType("preonly")
                self.ksp_fields[n].getPC().setType("lu")
                self.ksp_fields[n].getPC().setFactorSolverType("mumps")
            else:
                self.ksp_fields[n].setType(self.precond_fields[n]['solve'])
                self.ksp_fields[n].getPC().setType(self.precond_fields[n]['prec'])
                # raise ValueError("Unknown preconditioner type")

            self.ksp_fields[n].setOperators(operator_mats[n])

    def init_mat_vec(self, pc):
        self.A  = self.P.createSubMatrix(self.iset[0],self.iset[0])
        self.Bt = self.P.createSubMatrix(self.iset[0],self.iset[1])
        self.Dt = self.P.createSubMatrix(self.iset[0],self.iset[2])
        self.B  = self.P.createSubMatrix(self.iset[1],self.iset[0])
        self.C  = self.P.createSubMatrix(self.iset[1],self.iset[1])
        self.Et = self.P.createSubMatrix(self.iset[1],self.iset[2])
        self.D  = self.P.createSubMatrix(self.iset[2],self.iset[0])
        self.E  = self.P.createSubMatrix(self.iset[2],self.iset[1])
        self.R  = self.P.createSubMatrix(self.iset[2],self.iset[2])

        # the matrix to later insert the diagonal
        self.Adinv = PETSc.Mat().createAIJ(self.A.getSizes(), bsize=None, nnz=(1,1), csr=None, comm=self.comm)
        self.Adinv.setUp()
        self.Adinv.assemble()
        # set 1's to get correct allocation pattern
        self.Adinv.shift(1.)

        if self.schur_block_scaling[0]['type']=='diag':
            self.adinv_vec = self.A.getDiagonal()
        elif self.schur_block_scaling[0]['type']=='rowsum':
            self.adinv_vec = self.A.getRowSum()
        elif self.schur_block_scaling[0]['type']=='none':
            self.adinv_vec = self.A.createVecLeft()
            self.adinv_vec.set(1.0)
        else:
            raise ValueError("Unknown schur_block_scaling option!")

        self.Smod = self.C.copy(structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

        if self.schur_block_scaling[1]['type']=='diag':
            self.smoddinv_vec = self.Smod.getDiagonal()
        elif self.schur_block_scaling[1]['type']=='rowsum':
            self.smoddinv_vec = self.Smod.getRowSum()
        elif self.schur_block_scaling[1]['type']=='none':
            self.smoddinv_vec = self.Smod.createVecLeft()
            self.smoddinv_vec.set(1.0)
        else:
            raise ValueError("Unknown schur_block_scaling option!")

        # the matrix to later insert the diagonal
        self.Smoddinv = PETSc.Mat().createAIJ(self.C.getSizes(), bsize=None, nnz=(1,1), csr=None, comm=self.comm)
        self.Smoddinv.setUp()
        self.Smoddinv.assemble()
        # set 1's to get correct allocation pattern
        self.Smoddinv.shift(1.)

        self.Tmod = self.Et.copy(structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        self.Umod = self.E.copy(structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        self.Wmod = self.R.copy(structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

        self.Adinv_Bt = self.Adinv.matMult(self.Bt)
        self.D_Adinv_Bt = self.D.matMult(self.Adinv_Bt)

        self.B_Adinv_Bt = self.B.matMult(self.Adinv_Bt)

        self.Adinv_Dt = self.Adinv.matMult(self.Dt)
        self.B_Adinv_Dt = self.B.matMult(self.Adinv_Dt)

        self.D_Adinv_Dt = self.D.matMult(self.Adinv_Dt)

        # need to set Smod and Tmod here to get the data structures right
        self.Smod.axpy(-1., self.B_Adinv_Bt)
        self.Umod.axpy(-1., self.D_Adinv_Bt)
        self.Tmod.axpy(-1., self.B_Adinv_Dt)

        self.Smoddinv_Tmod = self.Smoddinv.matMult(self.Tmod)

        self.Umod_Smoddinv_Tmod = self.Umod.matMult(self.Smoddinv_Tmod)

        self.By1 = self.B.createVecLeft()
        self.Dy1 = self.D.createVecLeft()
        self.Umody2 = self.E.createVecLeft()
        self.Tmody3 = self.Et.createVecLeft()
        self.Bty2 = self.Bt.createVecLeft()
        self.Dty3 = self.Dt.createVecLeft()

        self.x1, self.x2, self.x3 = self.A.createVecLeft(), self.Smod.createVecLeft(), self.Wmod.createVecLeft()
        self.y1, self.y2, self.y3 = self.A.createVecLeft(), self.Smod.createVecLeft(), self.Wmod.createVecLeft()
        self.z1, self.z2, self.z3 = self.A.createVecLeft(), self.Smod.createVecLeft(), self.Wmod.createVecLeft()

        # do we need these???
        self.A.setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, True)
        self.Smod.setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, True)
        self.Wmod.setOption(PETSc.Mat.Option.NO_OFF_PROC_ZERO_ROWS, True)

        return [self.A, self.Smod, self.Wmod]

    def setUp(self, pc):
        self.P.createSubMatrix(self.iset[0],self.iset[0], submat=self.A)
        self.P.createSubMatrix(self.iset[0],self.iset[1], submat=self.Bt)
        self.P.createSubMatrix(self.iset[0],self.iset[2], submat=self.Dt)
        self.P.createSubMatrix(self.iset[1],self.iset[0], submat=self.B)
        self.P.createSubMatrix(self.iset[1],self.iset[1], submat=self.C)
        self.P.createSubMatrix(self.iset[1],self.iset[2], submat=self.Et)
        self.P.createSubMatrix(self.iset[2],self.iset[0], submat=self.D)
        self.P.createSubMatrix(self.iset[2],self.iset[1], submat=self.E)
        self.P.createSubMatrix(self.iset[2],self.iset[2], submat=self.R)

        if self.schur_block_scaling[0]['type']=='diag':
            self.A.getDiagonal(result=self.adinv_vec)
            self.adinv_vec.reciprocal()
        elif self.schur_block_scaling[0]['type']=='rowsum':
            self.A.getRowSum(result=self.adinv_vec)
            self.adinv_vec.abs()
            self.adinv_vec.reciprocal()
        elif self.schur_block_scaling[0]['type']=='none':
            self.adinv_vec.set(1.0)
        else:
            raise ValueError("Unknown schur_block_scaling option!")

        self.adinv_vec.scale(self.schur_block_scaling[0]['val'])

        # form diag(A)^{-1}
        self.Adinv.setDiagonal(self.adinv_vec, addv=PETSc.InsertMode.INSERT)

        self.Adinv.matMult(self.Bt, result=self.Adinv_Bt)      # diag(A)^{-1} Bt
        self.B.matMult(self.Adinv_Bt, result=self.B_Adinv_Bt)  # B diag(A)^{-1} Bt

        # --- modified Schur complement Smod = C - B diag(A)^{-1} Bt
        # compute self.Smod = self.C - B_Adinv_Bt
        self.C.copy(result=self.Smod)
        self.Smod.axpy(-1., self.B_Adinv_Bt)

        # --- Umod = E - D diag(A)^{-1} Bt
        # --- Tmod = Et - B diag(A)^{-1} Dt

        self.Adinv.matMult(self.Dt, result=self.Adinv_Dt)      # diag(A)^{-1} Dt
        self.B.matMult(self.Adinv_Dt, result=self.B_Adinv_Dt)  # B diag(A)^{-1} Dt
        self.D.matMult(self.Adinv_Bt, result=self.D_Adinv_Bt)  # D diag(A)^{-1} Bt

        # compute self.Umod = self.E - D_Adinv_Bt
        self.E.copy(result=self.Umod)
        self.Umod.axpy(-1., self.D_Adinv_Bt)

        # compute self.Tmod = self.Et - B_Adinv_Dt
        self.Et.copy(result=self.Tmod)
        self.Tmod.axpy(-1., self.B_Adinv_Dt)

        # --- Wmod = R - D diag(A)^{-1} Dt - Umod diag(Smod)^{-1} Tmod

        if self.schur_block_scaling[1]['type']=='diag':
            self.Smod.getDiagonal(result=self.smoddinv_vec)
            self.smoddinv_vec.reciprocal()
        elif self.schur_block_scaling[1]['type']=='rowsum':
            self.Smod.getRowSum(result=self.smoddinv_vec)
            self.smoddinv_vec.abs()
            self.smoddinv_vec.reciprocal()
        elif self.schur_block_scaling[1]['type']=='none':
            self.smoddinv_vec.set(1.0)
        else:
            raise ValueError("Unknown schur_block_scaling option!")

        self.smoddinv_vec.scale(self.schur_block_scaling[1]['val'])

        # form diag(Smod)^{-1}
        self.Smoddinv.setDiagonal(self.smoddinv_vec, addv=PETSc.InsertMode.INSERT)

        self.Smoddinv.matMult(self.Tmod, result=self.Smoddinv_Tmod)                        # diag(Smod)^{-1} Tmod

        self.Umod.matMult(self.Smoddinv_Tmod, result=self.Umod_Smoddinv_Tmod)              # Umod diag(Smod)^{-1} Tmod

        self.D.matMult(self.Adinv_Dt, result=self.D_Adinv_Dt)                              # D diag(A)^{-1} Dt

        # compute self.Wmod = self.R - D_Adinv_Dt - Umod_Smoddinv_Tmod
        self.R.copy(result=self.Wmod)
        self.Wmod.axpy(-1., self.D_Adinv_Dt)
        self.Wmod.axpy(-1., self.Umod_Smoddinv_Tmod)

        # operator values have changed - do we need to re-set them?
        self.ksp_fields[0].setOperators(self.A)
        self.ksp_fields[1].setOperators(self.Smod)
        self.ksp_fields[2].setOperators(self.Wmod)

    def apply(self, pc, x, y):

        # get subvectors (references!)
        x.getSubVector(self.iset[0], subvec=self.x1)
        x.getSubVector(self.iset[1], subvec=self.x2)
        x.getSubVector(self.iset[2], subvec=self.x3)

        tss = time.time()

        # 1) solve A * y_1 = x_1
        self.ksp_fields[0].solve(self.x1, self.y1)

        self.B.mult(self.y1, self.By1)

        # compute z2 = x2 - self.By1
        self.z2.axpby(1., 0., self.x2)
        self.z2.axpy(-1., self.By1)

        # 2) solve Smod * y_2 = z_2
        self.ksp_fields[1].solve(self.z2, self.y2)

        self.D.mult(self.y1, self.Dy1)
        self.Umod.mult(self.y2, self.Umody2)

        # compute z3 = x3 - self.Dy1 - self.Umody2
        self.z3.axpby(1., 0., self.x3)
        self.z3.axpy(-1., self.Dy1)
        self.z3.axpy(-1., self.Umody2)

        # 3) solve Wmod * y_3 = z_3
        self.ksp_fields[2].solve(self.z3, self.y3)

        self.Tmod.mult(self.y3, self.Tmody3)

        # compute z2 = x2 - self.By1 - self.Tmody3
        self.z2.axpy(-1., self.Tmody3)

        # 4) solve Smod * y_2 = z_2
        self.ksp_fields[1].solve(self.z2, self.y2)

        # 5) update y_1
        self.Adinv_Bt.mult(self.y2, self.Bty2)
        self.Adinv_Dt.mult(self.y3, self.Dty3)
        # compute y1 -= (self.Bty2 + self.Dty3)
        self.y1.axpy(-1., self.Bty2)
        self.y1.axpy(-1., self.Dty3)

        # restore/clean up
        x.restoreSubVector(self.iset[0], subvec=self.x1)
        x.restoreSubVector(self.iset[1], subvec=self.x2)
        x.restoreSubVector(self.iset[2], subvec=self.x3)

        # set into y vector
        y.setValues(self.iset[0], self.y1.array)
        y.setValues(self.iset[1], self.y2.array)
        y.setValues(self.iset[2], self.y3.array)

        y.assemble()

    def destroy(self, pc):
        pc.destroy()
