#!/usr/bin/env python3
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

import argparse
import gmsh
import meshio
import numpy as np
import subprocess
gmsh.initialize()


def read_spheres_position_file(spheres_position_path):
    """
    Reads file input that contains the centers of spheres. The path is assumed to have
    been generated using code from Skoge et al.
    """
    centers = []
    radius = 0
    n = 0
    with open(spheres_position_path) as fp:
        for i, row in enumerate(fp.readlines()):
            if i < 2:
                continue
            if i == 2:
                n = int(row)
            if i == 3:
                radius = float(row)
            if i < 6:
                continue
            x, y, z, _ = row.split(' ')
            centers.append((float(x), float(y), float(z)))
    return centers, float(radius)/2, n


def build_packed_spheres_mesh(output_mesh_file, spheres_locations_file):
    gmsh.model.add("3D")
    Lx, Ly, Lz = 1, 1, 1
    resolution = 0.01
    channel = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
    spheres_ = []
    centers, r, n_spheres = read_spheres_position_file(spheres_locations_file)
    for center in centers:
        x, y, z = center
        sphere = gmsh.model.occ.addSphere(*center, r)
        spheres_.append(sphere)
    channel = gmsh.model.occ.cut([(3, channel)], [(3, sphere) for sphere in spheres_])
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=3)
    marker = 11
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], marker)
    gmsh.model.setPhysicalName(volumes[0][0], marker, "conductor")
    surfaces = gmsh.model.occ.getEntities(dim=2)
    walls = []
    for surface in surfaces:
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        if np.isclose(com[1], 0):
            left_cc = gmsh.model.addPhysicalGroup(2, [surface[1]])
            gmsh.model.setPhysicalName(2, left_cc, "left_cc")
        elif np.isclose(com[1], Ly):
            right_cc = gmsh.model.addPhysicalGroup(2, [surface[1]])
            gmsh.model.setPhysicalName(2, right_cc, "right_cc")
        elif np.isclose(com[2], 0) or np.isclose(com[2], Lz) or np.isclose(com[0], 0) or np.isclose(com[0], Lx):
            walls.append(surface[1])
    insulated = gmsh.model.addPhysicalGroup(2, walls)
    gmsh.model.setPhysicalName(2, insulated, "insulated")

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "FacesList", [insulated, left_cc, right_cc])

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", resolution)
    gmsh.model.mesh.field.setNumber(2, "LcMax", 20*resolution)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.1)

    gmsh.model.mesh.field.add("Min", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    
    gmsh.write(output_mesh_file)
    
    return


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Effective Conductivity of Packed Spheres.')
    parser.add_argument('--pf', help='Packing fraction (%)', default=30)
    args = parser.parse_args()
    pf = args.pf
    grid_size = '1-1-1'
    spheres_locations_file = f"mesh/packed_spheres/{grid_size}_{pf}/{pf}.dat"
    output_mesh_file = f"mesh/packed_spheres/{grid_size}_{pf}/spheres.msh"
    
    build_packed_spheres_mesh(output_mesh_file, spheres_locations_file)
    mesh_3d = meshio.read(output_mesh_file)
    tetrahedral_mesh = create_mesh(mesh_3d, "tetra")
    meshio.write(f"mesh/packed_spheres/{grid_size}_{pf}/tetr.xdmf", tetrahedral_mesh)
    tria_mesh = create_mesh(mesh_3d, "triangle")
    meshio.write(f"mesh/packed_spheres/{grid_size}_{pf}/tria.xdmf", tria_mesh)
    transport_model_path = "transport.py"
    val = subprocess.check_call(f'mpirun -n 1 python3 {transport_model_path} --grid_size={grid_size} --data_dir=mesh/packed_spheres/{grid_size}_{pf}/', shell=True)
