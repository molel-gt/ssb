import os

import dolfinx
import gmsh
import numpy as np
import ufl

from dolfinx.fem import petsc
from dolfinx import cpp, fem, mesh, io, plot
from mpi4py import MPI
from petsc4py import PETSc

import pyvista as pv
import pyvistaqt as pvqt
have_pyvista = True
if pv.OFF_SCREEN:
    pv.start_xvfb(wait=0.5)

encoding = io.XDMFFile.Encoding.HDF5
markers = {
    'ne_pcc': 0,  # boundary not included yet
    'ne_se': 1,  # left boundary
    'pe_se': 2,  # internal boundary
    'pe_pcc': 3,  # left boundary
    'ne': 4,   # domain not included yet
    'se': 5,  # left domain
    'pe': 6,  # right domain
    "insulated_se": 7,  # top and bottom boundary - left
    "insulated_pam": 8,  # top and bottom boundary - right
}

# Define temporal parameters
t = 0 # Start time

dt = 5e-3
T = 50 * dt

workdir = "output/full-cell"
output_meshfile = os.path.join(workdir, 'mesh.msh')
tria_meshfile = os.path.join(workdir, "tria.xdmf")
line_meshfile = os.path.join(workdir, "line.xdmf")

comm = MPI.COMM_WORLD

# create submesh
full_mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(output_meshfile, comm, 0)

# Check physical volumes and surfaces of the mesh
print(np.unique(cell_tags.values))
print(np.unique(facet_tags.values))

# Create submesh for pe
domain, entity_map, vertex_map, geom_map = dolfinx.mesh.create_submesh(full_mesh, full_mesh.topology.dim, cell_tags.indices[(cell_tags.values == markers['pe'])])

# Transfer facet tags from parent mesh to submesh
tdim = full_mesh.topology.dim
fdim = tdim - 1
c_to_f = full_mesh.topology.connectivity(tdim, fdim)
f_map = full_mesh.topology.index_map(fdim)
all_facets = f_map.size_local + f_map.num_ghosts
all_values = np.zeros(all_facets, dtype=np.int32)
all_values[facet_tags.indices] = facet_tags.values
print(np.unique(all_values))

domain.topology.create_entities(fdim)
subf_map = domain.topology.index_map(fdim)
domain.topology.create_connectivity(tdim, fdim)
c_to_f_sub = domain.topology.connectivity(tdim, fdim)
num_sub_facets = subf_map.size_local + subf_map.num_ghosts
sub_values = np.empty(num_sub_facets, dtype=np.int32)
for i, entity in enumerate(entity_map):
    parent_facets = c_to_f.links(entity)
    child_facets = c_to_f_sub.links(i)
    for child, parent in zip(child_facets, parent_facets):
        sub_values[child] = all_values[parent]
sub_meshtag = dolfinx.mesh.meshtags(domain, domain.topology.dim - 1, np.arange(
    num_sub_facets, dtype=np.int32), sub_values)
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

with dolfinx.io.XDMFFile(comm, "submesh.xdmf", "w", encoding=encoding) as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(sub_meshtag, x=domain.geometry)

dx = ufl.Measure("dx", domain=domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=sub_meshtag)
dS = ufl.Measure("dS", domain=domain, subdomain_data=sub_meshtag)
tdim = domain.topology.dim
fdim = tdim - 1

boundary_facets = sub_meshtag.find(markers['pe_se'])

Q = fem.functionspace(domain, ("CG", 1))

# Create initial condition
c_n = fem.Function(Q)
c_n.name = "u_n"
c_n.interpolate(lambda x:  x[0] - x[0] + 1)

# Create boundary condition

bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(Q, fdim, boundary_facets), Q)

ch = fem.Function(Q)
ch.name = "concentration"
ch.interpolate(lambda x: x[0] - x[0] + 1)

c = ufl.TrialFunction(Q)
q = ufl.TestFunction(Q)

f = fem.Constant(domain, PETSc.ScalarType(0))
g = fem.Constant(domain, PETSc.ScalarType(0))

D = fem.Constant(domain, PETSc.ScalarType(1e-6))
flux = fem.Constant(domain, PETSc.ScalarType(1e-2))
a = c * q * dx(markers['pe']) + dt * ufl.inner(D * ufl.grad(c), ufl.grad(q)) * dx #(markers['pe'])
# L = (c_n + dt * f) * q * dx + dt * flux * q * ds(markers['pe_se']) + dt * ufl.inner(g, q) * ds(markers['insulated_pam']) + dt * ufl.inner(g, q) * ds(markers['pe_pcc'])#(markers['pe'])
L = (c_n + dt * f) * q * dx + dt * ufl.inner(g, q) * ds(markers['insulated_pam']) + dt * ufl.inner(g, q) * ds(markers['pe_pcc'])

bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = petsc.assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = fem.petsc.create_vector(linear_form)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

file = io.XDMFFile(comm, "output/full-cell/concentration.xdmf", "w", encoding=encoding)
file.write_mesh(domain)
file.write_function(c_n, 0)

# pyvista
topology, cell_types, x = plot.vtk_mesh(Q)
grid = pv.UnstructuredGrid(topology, cell_types, x)

# Set output data
grid.point_data["c"] = ch.x.array.real
grid.set_active_scalars("c")

p = pvqt.BackgroundPlotter(title="concentration", auto_update=True)
p.add_mesh(grid, clim=[0, 1])
p.view_xy(True)
p.add_text(f"time: {t}", font_size=12, name="timelabel")
p.add_axes()

while t < T:
    t += dt
    A = fem.petsc.assemble_matrix(fem.form(a), bcs=[bc])
    A.assemble()
    solver.setOperators(A)

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    fem.petsc.assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    fem.petsc.apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, [bc])

    # Solve linear problem
    solver.solve(b, ch.vector)
    ch.x.scatter_forward()

    # Update solution at previous time step (c_n)
    c_n.x.array[:] = ch.x.array
    file.write_function(ch, t)
    p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
    grid.point_data["c"] = ch.x.array.real
    p.app.processEvents()
file.close()
