import numbers
from dolfinx import cpp, io
from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py import PETSc

import geometry

def transfer_meshtags_to_submesh(
    domain, entity_tag, submesh, sub_vertex_to_parent, sub_cell_to_parent
):
    """
    Transfer a meshtag from a parent mesh to a sub-domain.
    """

    tdim = domain.topology.dim
    cell_imap = domain.topology.index_map(tdim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    mesh_to_submesh = np.full(num_cells, -1)
    mesh_to_submesh[sub_cell_to_parent] = np.arange(
        len(sub_cell_to_parent), dtype=np.int32
    )
    sub_vertex_to_parent = np.asarray(sub_vertex_to_parent)

    submesh.topology.create_connectivity(entity_tag.dim, 0)

    num_child_entities = (
        submesh.topology.index_map(entity_tag.dim).size_local
        + submesh.topology.index_map(entity_tag.dim).num_ghosts
    )
    submesh.topology.create_connectivity(submesh.topology.dim, entity_tag.dim)

    c_c_to_e = submesh.topology.connectivity(submesh.topology.dim, entity_tag.dim)
    c_e_to_v = submesh.topology.connectivity(entity_tag.dim, 0)

    child_markers = np.full(num_child_entities, 0, dtype=np.int32)

    domain.topology.create_connectivity(entity_tag.dim, 0)
    domain.topology.create_connectivity(entity_tag.dim, domain.topology.dim)
    p_f_to_v = domain.topology.connectivity(entity_tag.dim, 0)
    p_f_to_c = domain.topology.connectivity(entity_tag.dim, domain.topology.dim)
    sub_to_parent_entity_map = np.full(num_child_entities, -1, dtype=np.int32)
    for facet, value in zip(entity_tag.indices, entity_tag.values):
        facet_found = False
        for cell in p_f_to_c.links(facet):
            if facet_found:
                break
            if (child_cell := mesh_to_submesh[cell]) != -1:
                for child_facet in c_c_to_e.links(child_cell):
                    child_vertices = c_e_to_v.links(child_facet)
                    child_vertices_as_parent = sub_vertex_to_parent[child_vertices]
                    is_facet = np.isin(
                        child_vertices_as_parent, p_f_to_v.links(facet)
                    ).all()
                    if is_facet:
                        child_markers[child_facet] = value
                        facet_found = True
                        sub_to_parent_entity_map[child_facet] = facet
    tags = dolfinx.mesh.meshtags(
        submesh,
        entity_tag.dim,
        np.arange(num_child_entities, dtype=np.int32),
        child_markers,
    )
    tags.name = entity_tag.name

    return tags, sub_to_parent_entity_map


def transfer_meshtags(domain, submesh, entity_map, ft):
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
    c_to_f_sub = submesh.topology.connectivity(tdim, fdim)
    num_sub_facets = subf_map.size_local + subf_map.num_ghosts
    sub_values = np.empty(num_sub_facets, dtype=np.int32)
    for i, entity in enumerate(entity_map):
        parent_facets = c_to_f.links(entity)
        child_facets = c_to_f_sub.links(i)
        for child, parent in zip(child_facets, parent_facets):
            sub_values[child] = all_values[parent]
    submesh_ft = dolfinx.mesh.meshtags(submesh, submesh.topology.dim - 1, np.arange(
        num_sub_facets, dtype=np.int32), sub_values)
    submesh.topology.create_connectivity(submesh.topology.dim - 1, submesh.topology.dim)

    return submesh_ft



def compute_cell_boundary_facets(domain, ct, marker):
    """Compute the integration entities for integrals around the
    boundaries of all cells in domain.

    Parameters:
        domain: The mesh.
        ct: cell tags
        marker: physical group label

    Returns:
        Facets to integrate over, identified by ``(cell, local facet
        index)`` pairs.
    """
    tdim = domain.topology.dim
    fdim = tdim - 1
    n_f = cpp.mesh.cell_num_entities(domain.topology.cell_type, fdim)
    if isinstance(marker, numbers.Number):
        cells_1 = ct.find(marker)
    else:
        cells_1 = np.hstack([ct.find(m) for m in marker])
    perm = np.argsort(cells_1)
    n_c = cells_1.shape[0]

    return np.vstack((np.repeat(cells_1[perm], n_f), np.tile(np.arange(n_f), n_c))).T


def compute_interface_cell_boundary_facets(domain, ct, ft, cell_marker, facet_marker):
    """
    Compute integration entities for integrals around the boundaries of cells at the
    location of prescribed flux expression

    domain: the mesh
    ct: cell tags
    ft: facet tags
    cell_marker: marker for subdomain
    facet_marker: marker for interface
    """
    tdim = domain.topology.dim
    fdim = tdim - 1
    f_to_c = domain.topology.connectivity(fdim, tdim)
    c_to_f = domain.topology.connectivity(tdim, fdim)
    ft_imap = domain.topology.index_map(fdim)
    num_facets = ft_imap.size_local + ft_imap.num_ghosts
    interface_facets = ft.find(facet_marker)

    int_facet_domain = []
    lcells = []
    for f in interface_facets:
        if f >= ft_imap.size_local:
            continue
        c_0 = f_to_c.links(f)[0]
        subdomain_0 = ct.values[c_0]
        local_f_0 = np.where(c_to_f.links(c_0) == f)[0][0]
        if subdomain_0 == cell_marker:
            int_facet_domain.append([c_0, local_f_0])

        if len(f_to_c.links(f)) == 2:
            c_1 = f_to_c.links(f)[1]
            subdomain_1 = ct.values[c_1]
            if subdomain_1 == cell_marker:
                local_f_1 = np.where(c_to_f.links(c_1) == f)[0][0]
                int_facet_domain.append([c_1, local_f_1])

    return int_facet_domain


def xdmf_reader(mesh_folder, comm):
    """
    reads mesh_3d.xdmf and mesh_2d.xdmf from `mesh_folder`

    :return:
        domain, cell_tags, facet_tags
    """
    mesh_3d_file = os.path.join(mesh_folder, "mesh_3d.xdmf")
    mesh_2d_file = os.path.join(mesh_folder, "mesh_2d.xdmf")
    with io.XDMFFile(comm, mesh_3d_file, "r") as xdmf:
        domain = xdmf.read_mesh(cpp.mesh.GhostMode.shared_facet, name="Grid")
        ct = xdmf.read_meshtags(domain, name="Grid")

    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
    with io.XDMFFile(comm, mesh_2d_file, "r") as xdmf:
        ft = xdmf.read_meshtags(domain, name="Grid")

    return domain, ct, ft


def convert_to_xdmf(input_meshfile, cell_type_3d, cell_type_2d, key_name):
    """
    Extract 3D and 2D mesh from input file and convert to xdmf output file.
    """
    mesh = meshio.read(input_meshfile)
    mesh_folder = os.path.dirname(input_meshfile)
    mesh_3d_file = os.path.join(mesh_folder, "mesh_3d.xdmf")
    mesh_2d_file = os.path.join(mesh_folder, "mesh_2d.xdmf")
    mesh_3d = geometry.create_mesh(mesh, cell_type_3d, key_name=key_name)
    mesh_3d.write(mesh_3d_file)
    mesh_2d = geometry.create_mesh(mesh, cell_type_2d, key_name=key_name)
    mesh_2d.write(mesh_2d_file)

    return
