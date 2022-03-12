#!/usr/bin/env python3

import os

import argparse
import matplotlib.pyplot as plt
import meshio
import numpy as np
import subprocess


def load_images_to_logical_array(files_list, x_lims=(0, 201), y_lims=(0, 201), z_lims=(0, 201)):
    """
    grid_sizes: Lx.Ly.Lz
    """
    x0, x1 = x_lims
    y0, y1 = y_lims
    z0, z1 = z_lims
    Lx = x1 - x0
    Ly = y1 - y0
    Lz = z1 - z0
    data = np.zeros([int(Lx), int(Ly), int(Lz)], dtype=bool)
    for i_x, img_file in enumerate(files_list):
        if not (x0 <= i_x <= x1):
            continue
        img_data = plt.imread(img_file)
        img_data = img_data / 255
        data[i_x - x0 - 1, :, :] = img_data[y0:y1, z0:z1]
    return data


def compute_boundary_markers(local_pos, grid_shape):
    """"""
    # TODO: determine whether the position is at the faces of the box
    x, y, z = local_pos
    if x == 0:
        return 1
    elif y == 0:
        return 4
    elif z == 0:
        return 5
    elif x == grid_shape[0] - 1:
        return 3
    elif y == grid_shape[1] - 1:
        return 2
    elif z == grid_shape[2] - 1:
        return 6
    return 0


def create_nodes(data, **kwargs):
    """"""
    n_nodes = int(np.sum(data))
    nodes = np.zeros([n_nodes, 4])
    count = 0
    for idx, point in np.ndenumerate(data):
        if point:
            boundary_marker = compute_boundary_markers(idx, data.shape)
            nodes[count, :] = list(idx) + [boundary_marker]
            count += 1
    return nodes


def write_node_to_file(nodes, node_file_path):
    """"""
    count, _ = nodes.shape
    meta_header = "# Node count, 3 dim, no attribute, no boundary marker"
    header_data = [str(count), '3', '0', '1']
    entries_header = "# Node index, node coordinates, boundary marker"
    with open(node_file_path, "w") as fp:
        fp.write(meta_header + '\n')
        fp.write(' '.join(header_data) + '\n')
        fp.write(entries_header + '\n')
        for idx in range(count):
            entry = [idx] + list(nodes[idx, :])
            fp.write(' '.join([str(v) for v in entry]) + '\n')
    return


def create_mesh(mesh, cell_type, prune_z=False):
    """
    """
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points,
                           cells={cell_type: cells},
                           cell_data={"name_to_read": [cell_data]}
                           )
    if prune_z:
        out_mesh.prune_z_0()
    return out_mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build geometry')
    parser.add_argument('--img_sub_dir', help='bmp files sub directory', required=True)
    parser.add_argument('--grid_info', help='Nx-Ny-Nz', required=True)

    args = parser.parse_args()
    grid_info = args.grid_info
    Nx, Ny, Nz = grid_info.split("-")
    
    files_list = sorted([os.path.join(args.img_sub_dir, f) for f in os.listdir(args.img_sub_dir)
                  if f.endswith(".bmp")])
    
    meshes_dir = 'mesh'
    node_file_path = os.path.join(meshes_dir, '{}.node'.format(grid_info))
    geo_file_path = os.path.join(meshes_dir, '{}.geo'.format(grid_info))
    vtk_file_path = os.path.join(meshes_dir, '{}.vtk'.format(grid_info))
    msh_file_path = os.path.join(meshes_dir, '{}.msh'.format(grid_info))
    line_mesh_path = os.path.join(meshes_dir, f"{grid_info}_line.xdmf")
    tria_mesh_path = os.path.join(meshes_dir, f"{grid_info}_tria.xdmf")
    tetr_mesh_path = os.path.join(meshes_dir, f"{grid_info}_tetr.xdmf")

    image_data = load_images_to_logical_array(files_list, (0, int(Nx)), (0, int(Ny)), (0, int(Nz)))
    nodes = create_nodes(image_data)
    write_node_to_file(nodes, node_file_path)
    # build .msh file from .node file
    val = subprocess.check_call("./nodes_to_msh.sh %s %s %s %s" % (node_file_path, geo_file_path, vtk_file_path, msh_file_path), shell=True)
    
    # build .xdmf/.h5 file from .msh file
    msh = meshio.read(msh_file_path)
    print("creating tetrahedral mesh")  
    tetra_mesh = create_mesh(msh, "tetra")
    meshio.write(tetr_mesh_path, tetra_mesh)
    print("creating triangle mesh")
    triangle_mesh = create_mesh(msh, "triangle")
    meshio.write(tria_mesh_path, triangle_mesh)
    print("create line mesh")
    line_mesh = create_mesh(msh, "line")
    meshio.write(line_mesh_path, line_mesh)
    print("wrote files {}, {}, {}".format(tetr_mesh_path, tria_mesh_path, line_mesh_path))
