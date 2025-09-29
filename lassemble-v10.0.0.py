# Local assembly of cell integral or exterior facet integrals
# Author: Jørgen S. Dokken
# SPDX-License-Identifier: MIT

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import  dx, grad, inner, ds
from dolfinx import fem, mesh
import cffi
import numba
from ffcx.codegeneration.utils import get_void_pointer

_type_to_offset_index = {fem.IntegralType.cell: 0, fem.IntegralType.exterior_facet: 1}

ffi = cffi.FFI()

class LocalAssembler():

    def __init__(self, form, integral_type: fem.IntegralType =fem.IntegralType.cell):
        self.form = fem.form(form)
        self.integral_type = integral_type
        self.update_coefficients()
        self.update_constants()

        # subdomain_ids = self.form.integral_ids(integral_type)
        # assert(len(subdomain_ids) == 1)
        # assert(subdomain_ids[0] == -1)
        is_complex = np.issubdtype(ScalarType, np.complexfloating)
        nptype = "complex128" if is_complex else "float64"
        o_s = self.form.ufcx_form.form_integral_offsets[_type_to_offset_index[integral_type]]
        o_e = self.form.ufcx_form.form_integral_offsets[_type_to_offset_index[integral_type]+1]
        assert o_e - o_s == 1

        self.kernel = getattr(self.form.ufcx_form.form_integrals[o_s], f"tabulate_tensor_{nptype}")
        self.active_cells = self.form._cpp_object.domains(integral_type, 0)
        assert len(self.form.function_spaces) == 2
        self.local_shape = [0,0]
        for i, V in enumerate(self.form.function_spaces):
            self.local_shape[i] = V.dofmap.dof_layout.block_size * V.dofmap.dof_layout.num_dofs

        e0 = self.form.function_spaces[0].element
        e1 = self.form.function_spaces[1].element
        needs_transformation_data = e0.needs_dof_transformations or e1.needs_dof_transformations or \
            self.form._cpp_object.needs_facet_permutations
        if needs_transformation_data:
            raise NotImplementedError("Dof transformations not implemented")

        self.ffi = cffi.FFI()
        V = self.form.function_spaces[0]
        self.x_dofs = V.mesh.geometry.dofmap

    def update_coefficients(self):
        self.coeffs = fem.assemble.pack_coefficients(self.form)[(self.integral_type, 0)]

    def update_constants(self):
        self.consts = fem.assemble.pack_constants(self.form)

    def update(self):
        self.update_coefficients()
        self.update_constants()


@numba.njit(fastmath=True)
def assemble_matrix(kernel, mesh, local_shape, idx=2):
    x_dofs, x = mesh
    geometry = np.zeros((len(x_dofs[idx]), 3), dtype=x.dtype)
    geometry[:, :] = x[x_dofs[idx]]

    A_local = np.zeros((local_shape[0], local_shape[0]), dtype=ScalarType)
    facet_index = np.array([0], dtype=np.intc)
    facet_perm = np.zeros(0, dtype=np.uint8)
    coeffs = np.zeros(0, dtype=ScalarType)
    consts = np.zeros(1, dtype=ScalarType)
    custom_data = np.zeros(1, dtype=np.int64)
    custom_data_ptr = get_void_pointer(custom_data)
    ffi_fb = ffi.from_buffer
    kernel(ffi_fb(A_local), ffi_fb(coeffs), ffi_fb(consts), ffi_fb(geometry),
           ffi_fb(facet_index), ffi_fb(facet_perm), custom_data_ptr)
    return A_local


msh = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3,
                 mesh.CellType.triangle)

V = fem.functionspace(msh, ("Lagrange", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

assembler = LocalAssembler(inner(grad(u), grad(v)) * dx)
kernel = assembler.kernel
local_shape = assembler.local_shape
x_dofs = V.mesh.geometry.dofmap
x = msh.geometry.x

for cell in range(msh.topology.index_map(msh.topology.dim).size_local):
    if cell == 2:
        A = assemble_matrix(kernel, (x_dofs, x), local_shape)
        print(A)


assembler = LocalAssembler(inner(u, v) * ds, fem.IntegralType.exterior_facet)
kernel = assembler.kernel
local_shape = assembler.local_shape
x_dofs = V.mesh.geometry.dofmap
x = msh.geometry.x

for cell in range(msh.topology.index_map(msh.topology.dim).size_local):
    if cell == 2:
        A = assemble_matrix(kernel, (x_dofs, x), local_shape)
        print(A)
