#!/usr/bin/env python3

import os

import argparse
import matplotlib.pyplot as plt
import meshio
import numpy as np
import subprocess

import utils


def load_images_to_voxel(files_list, x_lims=(0, 201), y_lims=(0, 201), z_lims=(0, 201), origin=(0, 0, 0)):
    """
    grid_sizes: Lx.Ly.Lz
    """
    x0, x1 = x_lims
    y0, y1 = y_lims
    z0, z1 = z_lims
    Lx = x1 - x0
    Ly = y1 - y0
    Lz = z1 - z0
    x_shift, y_shift, z_shift = origin
    data = np.zeros([int(Lx), int(Ly), int(Lz)], dtype=bool)
    for i_x, img_file in enumerate(files_list):
        if not (x0 + x_shift <= i_x <= x1 + x_shift):
            continue
        img_data = plt.imread(img_file)
        img_data = img_data / 255
        data[i_x - x0 - x_shift - 1, :, :] = img_data[int(y_shift + y0):int(y1 + y_shift), int(z_shift + z0):int(z_shift + z1)]
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
    parser.add_argument('--img_folder', help='bmp files sub directory', required=True)
    parser.add_argument('--grid_info', help='Nx-Ny-Nz', required=True)
    parser.add_argument('--origin', default=(0, 0, 0), help='where to extract grid from')

    args = parser.parse_args()
    if isinstance(args.origin, str):
        origin = tuple(map(lambda v: int(v), args.origin.split(",")))
    else:
        origin = args.origin
    origin_str = "_".join([str(v) for v in origin])
    grid_info = args.grid_info
    Nx, Ny, Nz = grid_info.split("-")
    
    files_list = sorted([os.path.join(args.img_folder, f) for f in os.listdir(args.img_folder)
                  if f.endswith(".bmp")])
    
    meshes_dir = 'mesh'
    utils.make_dir_if_missing(meshes_dir)
    node_file_path = os.path.join(meshes_dir, f's{grid_info}o{origin_str}.node')
    geo_file_path = os.path.join(meshes_dir, f's{grid_info}o{origin_str}.geo')
    vtk_file_path = os.path.join(meshes_dir, f's{grid_info}o{origin_str}.vtk')
    msh_file_path = os.path.join(meshes_dir, f's{grid_info}o{origin_str}.msh')
    line_mesh_path = os.path.join(meshes_dir, f"s{grid_info}o{origin_str}_line.xdmf")
    tria_mesh_path = os.path.join(meshes_dir, f"s{grid_info}o{origin_str}_tria.xdmf")
    tetr_mesh_path = os.path.join(meshes_dir, f"s{grid_info}o{origin_str}_tetr.xdmf")

    image_data = load_images_to_voxel(files_list, (0, int(Nx)), (0, int(Ny)), (0, int(Nz)), origin)
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
    print(f"wrote files {tetr_mesh_path}, {tria_mesh_path}, {line_mesh_path}")
