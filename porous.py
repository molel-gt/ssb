#!/usr/bin/env python3
# coding: utf-8

import os
import warnings

import argparse
import gmsh
import meshio
import numpy as np

from stl import mesh

import geometry
import particles


gmsh.initialize()
warnings.filterwarnings("ignore")


def get_local_faces(point, surface_points):
    """
    :param surface_points: `list`
    :param point: `tuple(x, y, z)`
    """
    faces = set()
    point_idx = surface_points[point]
    x0, y0, z0 = point
    pairs = [
        [(int(x0 + 1), y0, z0), (x0, int(y0 + 1), z0)],
        # [(int(x0 + 1), y0, z0), (x0, int(y0 - 1), z0)],
        # [(int(x0 - 1), y0, z0), (x0, int(y0 + 1), z0)],
        [(int(x0 - 1), y0, z0), (x0, int(y0 - 1), z0)],

        [(int(x0 + 1), y0, z0), (x0, y0, int(z0 + 1))],
        # [(int(x0 + 1), y0, z0), (x0, y0, int(z0 - 1))],
        # [(int(x0 - 1), y0, z0), (x0, y0, int(z0 + 1))],
        [(int(x0 - 1), y0, z0), (x0, y0, int(z0 - 1))],

        [(x0, y0, int(z0 + 1)), (x0, int(y0 + 1), z0)],
        # [(x0, y0, int(z0 + 1)), (x0, int(y0 - 1), z0)],
        # [(x0, y0, int(z0 - 1)), (x0, int(y0 + 1), z0)],
        [(x0, y0, int(z0 - 1)), (x0, int(y0 - 1), z0)],
    ]
    for point_1, point_2 in pairs:
        coord1_idx = surface_points.get(point_1)
        coord2_idx = surface_points.get(point_2)
        if coord1_idx is not None and coord2_idx is not None:
            f = (point_idx, coord1_idx, coord2_idx)
            faces.add(f)
    return faces


def porous_mesh_surface(surface_points, stl_file):
    """"""
    vertices = np.array([list(k) for k, _ in surface_points.items()])

    # face triangles
    faces = set()
    for k, _ in surface_points.items():
        local_faces = get_local_faces(k, surface_points)
        for f in local_faces:
            faces.add(f)
    faces = np.array(list(faces))

    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[face[j], :]

    # Write the mesh to file
    cube.save(f'{stl_file}')


def porous_mesh_volume(stl_file, msh_file):
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.QualityType", 2)

    gmsh.merge(f"{stl_file}")
    n = gmsh.model.getDimension()
    s = gmsh.model.getEntities(n)
    l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
    gmsh.model.geo.addVolume([l])
    print("Volume added")
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(3)
    volumes = gmsh.model.getEntities(dim=3)
    marker = 11
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], marker)
    gmsh.write(f"{msh_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='computes specific area')
    parser.add_argument('--img_folder', help='bmp files directory',
                        required=True)
    parser.add_argument('--grid_info', help='Nx-Ny-Nz',
                        required=True)
    parser.add_argument('--origin', default=(0, 0, 0), help='where to extract grid from')

    args = parser.parse_args()
    if isinstance(args.origin, str):
        origin = tuple(map(lambda v: int(v), args.origin.split(",")))
    elif isinstance(args.origin, tuple):
        origin = args.origin
    origin_str = "_".join([str(v) for v in origin])
    grid_info = args.grid_info
    grid_size = int(args.grid_info.split("-")[0])
    Nx, Ny, Nz = [int(v) for v in args.grid_info.split("-")]
    img_dir = args.img_folder
    im_files = sorted([os.path.join(img_dir, f) for
                       f in os.listdir(img_dir) if f.endswith(".bmp")])
    n_files = len(im_files)

    data = geometry.load_images_to_voxel(im_files, x_lims=(0, Nx),
                                         y_lims=(0, Ny), z_lims=(0, Nz), origin=origin)
    home_dir = os.environ["HOME"]
    task_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../')
    output_mesh_file = os.path.join(task_dir, "mesh/porous.msh")

    origin_str = "_".join(map(lambda v: str(v), origin))
    _, surface_points = particles.filter_interior_points(data)

    porous_mesh_surface(surface_points, "porous.stl")
    porous_mesh_volume("porous.stl", "porous.msh")
    print("writing xmdf tetrahedral mesh..")
    msh = meshio.read("porous.msh")
    tetra_mesh = geometry.create_mesh(msh, "tetra")
    meshio.write(f"mesh/s{grid_info}o{origin_str}_tetr.xdmf", tetra_mesh)

    gmsh.finalize()