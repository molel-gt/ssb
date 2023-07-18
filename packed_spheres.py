#!/usr/bin/env python3
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

import argparse
import gmsh
import meshio
import numpy as np

import commons, geometry, utils


cell_types = commons.CellTypes()
markers = commons.SurfaceMarkers()

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
    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1e-4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1e-2)
    gmsh.model.add("3D")
    Lx, Ly, Lz = 1, 1, 1
    resolution = 1e-6
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
    left_surfs = []
    right_surfs = []
    for surface in surfaces:
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        if np.isclose(com[2], 0):
            left_surfs.append(surface[1])
        elif np.isclose(com[2], Lz):
            right_surfs.append(surface[1])
        elif np.isclose(com[1], 0) or np.isclose(com[1], Ly) or np.isclose(com[0], 0) or np.isclose(com[0], Lx):
            walls.append(surface[1])
    left_cc = gmsh.model.addPhysicalGroup(2, left_surfs, markers.left_cc)
    gmsh.model.setPhysicalName(2, left_cc, "left_cc")
    right_cc = gmsh.model.addPhysicalGroup(2, right_surfs, markers.right_cc)
    gmsh.model.setPhysicalName(2, right_cc, "right_cc")
    insulated = gmsh.model.addPhysicalGroup(2, walls, markers.insulated)
    gmsh.model.setPhysicalName(2, insulated, "insulated")

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "FacesList", [insulated, left_cc, right_cc])

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", resolution)
    gmsh.model.mesh.field.setNumber(2, "LcMax", 20*resolution)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.1)

    # gmsh.model.mesh.field.add("Min", 5)
    # gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
    # gmsh.model.mesh.field.setAsBackgroundMesh(5)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    
    gmsh.write(output_mesh_file)
    gmsh.finalize()
    
    return


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Packed Spheres.')
    parser.add_argument('--packing_fraction', help="Packing fraction (percent)", default=0.3)
    args = parser.parse_args()
    scale_factor = [25e-6, 25e-6, 25e-6]  # microns
    outdir = f'mesh/packed_spheres/2-2-2/{args.packing_fraction}'
    spheres_locations_file = f"{outdir}/centers.dat"
    output_mesh_file = f"{outdir}/spheres.msh"
    
    build_packed_spheres_mesh(output_mesh_file, spheres_locations_file)
    mesh_3d = meshio.read(output_mesh_file)
    tetrahedral_mesh = create_mesh(mesh_3d, "tetra")
    meshio.write(f"{outdir}/tetr.xdmf", tetrahedral_mesh)
    tetr_mesh_scaled = geometry.scale_mesh(tetrahedral_mesh, cell_types.tetra, scale_factor=scale_factor)
    tetr_mesh_scaled.write(f"{outdir}/tetr.xdmf")
    tria_mesh = create_mesh(mesh_3d, "triangle")
    meshio.write(f"{outdir}/tria.xdmf", tria_mesh)
    tria_mesh_scaled = geometry.scale_mesh(tria_mesh, cell_types.triangle, scale_factor=scale_factor)
    tria_mesh_scaled.write(f"{outdir}/tria.xdmf")

    print(f"Wrote tetr.xdmf and tria.xdmf mesh files to directory: {outdir}")
