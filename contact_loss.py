#!/usr/bin/env python3

import pickle
import subprocess
import sys

import argparse
import gmsh
import meshio
import numpy as np

from skimage import io
from sympy import arg

import connected_pieces, geometry, utils, volume_reconstruction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Volume with Contact Loss")
    parser.add_argument("--grid_info", help="Nx-Ny-Nz that defines the grid size", required=True)
    parser.add_argument("--contact_map_file", help="Image to generate contact map", required=True)
    parser.add_argument("--phase", help="0 -> void, 1 -> SE, 2 -> AM", nargs='?', const=1, default=1)
    args = parser.parse_args()
    grid_info = args.grid_info
    contact_img_file = args.contact_map_file
    phase = args.phase
    scale_factor = (1.0, 1.0, 1.0)
    origin_str = 'current-constriction'
    Nx, Ny, Nz = [int(v) for v in grid_info.split("-")]
    utils.make_dir_if_missing(f"mesh/{phase}/{grid_info}_{origin_str}")
    nodefile = f"mesh/{phase}/{grid_info}_{origin_str}/porous.node"
    tetfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous.ele"
    facesfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous.face"
    vtkfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous.1.vtk"
    surface_vtk = f"mesh/{phase}/{grid_info}_{origin_str}/surface.vtk"
    tetr_mshfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous_tetr.msh"
    surf_mshfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous_tria.msh"
    tetr_xdmf_scaled = f"mesh/{phase}/{grid_info}_{origin_str}/tetr.xdmf"
    tetr_xdmf_unscaled = f"mesh/{phase}/{grid_info}_{origin_str}/tetr_unscaled.xdmf"
    tria_xdmf_scaled = f"mesh/{phase}/{grid_info}_{origin_str}/tria.xdmf"
    tria_xdmf_unscaled = f"mesh/{phase}/{grid_info}_{origin_str}/tria_unscaled.xdmf"
    tria_xmf_unscaled = f"mesh/{phase}/{grid_info}_{origin_str}/tria_unscaled.xmf"

    img = io.imread(contact_img_file)
    contact_points = set()
    for idx in np.argwhere(np.isclose(img, phase)):
        contact_points.add(tuple([int(v) for v in idx] + [0]))

    contact_points_filepath = f"mesh/{phase}/{grid_info}_{origin_str}/contact_points.pickle"
    with open(contact_points_filepath, "wb") as fp:
        pickle.dump(contact_points, fp, protocol=pickle.HIGHEST_PROTOCOL)

    box = np.ones((Nx, Ny, Nz), dtype=np.uint8)
    points = connected_pieces.build_points(box)
    cubes = volume_reconstruction.build_cubes(box, points)
    tetrahedra = {}
    n_tetrahedra = 0

    for cube in cubes:
        _tetrahedra = volume_reconstruction.build_tetrahedra(cube, points)
        for tet in _tetrahedra:
            tetrahedra[tet] = n_tetrahedra
            n_tetrahedra += 1

    with open(nodefile, "w") as fp:
        fp.write("%d 3 0 0\n" % int(len(points.values())))
        for coord, point_id in points.items():
            x0, y0, z0 = coord
            fp.write(f"{point_id} {x0} {y0} {z0}\n")
    tet_points = set()
    with open(tetfile, "w") as fp:
        fp.write(f"{n_tetrahedra} 4 0\n")
        for tetrahedron, tet_id in tetrahedra.items():
            p1, p2, p3, p4 = [int(v) for v in tetrahedron]
            tet_points |= {p1, p2, p3, p4}
            fp.write(f"{tet_id} {p1} {p2} {p3} {p4}\n")

    retcode_tetgen = subprocess.check_call(f"tetgen {tetfile} -rkQF", shell=True)
    gmsh.initialize()

    gmsh.model.add("porous")
    gmsh.merge(vtkfile)
    gmsh.model.occ.synchronize()
    counter = 0
    volumes = gmsh.model.getEntities(dim=3)
    for i, volume in enumerate(volumes):
        marker = int(counter + i)
        gmsh.model.addPhysicalGroup(3, [volume[1]], marker)
        gmsh.model.setPhysicalName(3, marker, f"V{marker}")
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(tetr_mshfile)
    gmsh.finalize()

    tet_msh = meshio.read(tetr_mshfile)
    tetr_mesh_unscaled = geometry.create_mesh(tet_msh, "tetra")
    tetr_mesh_unscaled.write(tetr_xdmf_unscaled)
    tetr_mesh_scaled = volume_reconstruction.scale_mesh(tetr_mesh_unscaled, "tetra", scale_factor=scale_factor)
    tetr_mesh_scaled.write(tetr_xdmf_scaled)