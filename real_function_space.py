
import gmsh
from packaging.version import Version
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh
from dolfinx.io import gmshio, VTXWriter
from dolfinx.cpp.la.petsc import scatter_local_vectors, get_local_vectors
import dolfinx.fem.petsc

import numpy as np
from scifem import create_real_functionspace, assemble_scalar
import ufl

import mesh_utils, solvers, utils


class Boundaries:
    def __init__(self):
        pass

    @property
    def left(self):
        return 1

    @property
    def bottom(self):
        return 2

    @property
    def right(self):
        return 3

    @property
    def top(self):
        return 4

    @property
    def insulated(self):
        return 5

    @property
    def domain(self):
        return 5


def build_mesh(output_path, markers, Lx=10, Ly=1):
    """
    generate mesh for given dimensions (`Lx` `Ly`) and write output to `output_path`
    """
    gmsh.initialize()
    gmsh.model.add('2D')
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)
    coords = [
    (0, 0, 0),
    (Lx, 0, 0),
    (Lx, Ly, 0),
    (0, Ly, 0),
    ]
    points = []
    lines = []
    # adding coordinates
    for coord in coords:
        points.append(gmsh.model.occ.addPoint(*coord))

    # adding lines
    for idx in range(-1, len(points)-1):
        lines.append(gmsh.model.occ.addLine(points[idx], points[idx+1]))

    gmsh.model.occ.synchronize()

    # connect lines into loop
    loop = gmsh.model.occ.addCurveLoop(lines)

    # create surface from loop
    surf = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()

    # add boundary markers
    gmsh.model.addPhysicalGroup(1, [lines[0]], markers.left, "left")
    gmsh.model.addPhysicalGroup(1, [lines[1]], markers.bottom, "bottom")
    gmsh.model.addPhysicalGroup(1, [lines[2]], markers.right, "right")
    gmsh.model.addPhysicalGroup(1, [lines[3]], markers.top, "top")
    gmsh.model.addPhysicalGroup(1, [lines[1], lines[3]], markers.insulated, "insulated")
    gmsh.model.addPhysicalGroup(2, [surf], markers.domain, "domain")
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(output_path)
    gmsh.finalize()

    return 0


if __name__ == '__main__':
    markers = Boundaries()
    output_mesh_path = 'mesh.msh'
    output_potential_path = 'potential.bp'
    output_current_path = 'current.bp'
    # build_mesh(output_mesh_path, markers, Lx=1)

    comm = MPI.COMM_WORLD
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    domain, ct, ft = gmshio.read_from_msh(output_mesh_path, comm, partitioner=partitioner)[:3]
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)

    left_boundary = ft.find(markers.left)
    bottom_boundary = ft.find(markers.bottom)
    right_boundary = ft.find(markers.right)
    top_boundary = ft.find(markers.top)

    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)

    V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

    x = ufl.SpatialCoordinate(domain)
    n = ufl.FacetNormal(domain)

    # facets submesh
    submesh_facets, submesh_facets_to_mesh, f_v_map = mesh.create_submesh(
        domain, fdim, ft.indices)[:3]
    num_facets_local = (
        domain.topology.index_map(fdim).size_local + domain.topology.index_map(fdim).num_ghosts
    )
    parent_to_facets = np.full(num_facets_local, -1, dtype=np.int32)
    parent_to_facets[submesh_facets_to_mesh] = np.arange(len(submesh_facets_to_mesh), dtype=np.int32)
    entity_maps = {submesh_facets: parent_to_facets}

    all_facets = mesh_utils.compute_cell_boundary_facets(domain, ct, markers.domain)
    right_facets = mesh_utils.compute_interface_cell_boundary_facets(domain, ct, ft, markers.domain, markers.right)

    minus_right_facets = utils.delete_numpy_rows(all_facets, right_facets)
    right_bndry_facets = np.array(right_facets).flatten()

    # # Create the measure
    dx = ufl.Measure('dx', domain=domain, subdomain_data=ct, subdomain_id=markers.domain)
    ds_c = ufl.Measure("ds", subdomain_data=[(1, minus_right_facets.flatten()), (2, right_bndry_facets)], domain=domain)

    R = create_real_functionspace(submesh_facets)
    h = fem.Constant(submesh_facets, PETSc.ScalarType(1.0))

    u_left = fem.Function(V)
    with u_left.x.petsc_vec.localForm() as u0_loc:
        u0_loc.set(1.0)
    left_dofs = fem.locate_dofs_topological(V, 1, left_boundary)
    left_bc = fem.dirichletbc(u_left, left_dofs)

    u, lmbda = fem.Function(V), fem.Function(R)
    du, dl = ufl.TestFunction(V), ufl.TestFunction(R)

    zero = dolfinx.fem.Constant(submesh_facets, dolfinx.default_scalar_type(0.0))

    a00 = ufl.inner(ufl.grad(u), ufl.grad(du)) * dx
    L0 = ufl.inner(ufl.grad(du), n) * lmbda * ds_c(2)
    L1 = ufl.inner(zero, dl) * ds_c(2) 
    L1 += ufl.inner(ufl.grad(u), n) * dl * ds_c(2)

    a = dolfinx.fem.form([[a00, None], [None, None]], entity_maps=entity_maps)
    L = dolfinx.fem.form([L0, L1], entity_maps=entity_maps)
    maps = [(Wi.dofmap.index_map, Wi.dofmap.index_map_bs) for Wi in [V, R]]

    F0 = a00 + L0
    F1 = L1

    F = [fem.form(F0, entity_maps=entity_maps), fem.form(F1, entity_maps=entity_maps)]
    j00 = ufl.derivative(F0, u)
    j01 = ufl.derivative(F0, lmbda)
    j10 = ufl.derivative(F1, u)
    j11 = ufl.derivative(F1, lmbda)

    J00 = fem.form(j00, entity_maps=entity_maps)
    J01 = fem.form(j01, entity_maps=entity_maps)
    J10 = fem.form(j10, entity_maps=entity_maps)
    J11 = fem.form(j11, entity_maps=entity_maps)

    J = [[J00, J01], [J10, J11]]

    opts = {
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                'pc_factor_mat_solver_type': 'mumps',
                }

    solver = solvers.NewtonSolver(
                F,
                J,
                [u, lmbda],
                bcs=[left_bc],
                max_iterations=10,
                petsc_options=opts,
                maps=maps,
                h=h
                )
    solver.solve()

    current_l = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(ufl.inner(-ufl.grad(u), n)) * ds(markers.left))), op=MPI.SUM)
    current_r = domain.comm.allreduce(fem.assemble_scalar(fem.form(np.abs(ufl.inner(-ufl.grad(u), n)) * ds(markers.right))), op=MPI.SUM)
    print(current_l, current_r)

    with VTXWriter(comm, "potential.bp", [u], engine="BP4") as vtx:
        vtx.write(0.0)
