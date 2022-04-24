#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import argparse
import gmsh
import meshio
import numpy as np
import subprocess

import geometry


def make_rectangles(voxels):
    Nx, Ny, Nz = voxels.shape
    rectangles = np.zeros(voxels.shape, dtype=np.uint8)

    for idx in range(Nx - 1):
        for idy in range(Ny - 1):
            for idz in range(Nz):
                p0 = (idx, idy, idz)
                p1 = (idx + 1, idy, idz)
                p2 = (idx, idy + 1, idz)
                p3 = (idx + 1, idy + 1, idz)

                if voxels[p0] and voxels[p1] and voxels[p2] and voxels[p3]:
                    rectangles[p0] = 1

    return rectangles


def make_boxes(rectangles):
    Nx, Ny, Nz = rectangles.shape
    boxes = np.zeros(rectangles.shape, dtype=np.uint16)
    for idx in range(Nx - 1):
        for idy in range(Ny - 1):
            box_length = 0
            start_pos = 0
            counter = 0
            for idz in range(Nz - 1):
                if not rectangles[idx][idy][idz]:
                    if box_length > 0:
                        boxes[idx][idz][start_pos] = box_length
                        counter += 1
                    start_pos = idz
                    box_length = 0
                    continue
                if rectangles[idx][idy][idz] and rectangles[idx][idy][idz + 1]:
                    box_length += 1
                    if idz == (Nx - 3):
                        boxes[idx][idy][start_pos] = box_length
                    start_pos = idz
    return boxes


def build_voxels_mesh(boxes, output_mshfile):
    gmsh.model.add("3D")
    Nx, Ny, Nz = boxes.shape
    Lx = Nx - 1
    Ly = Ny - 1
    Lz = Nz - 1
    resolution = 0.1
    channel = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
    gmsh_boxes = []

    for index, box_length in np.ndenumerate(boxes):
        if box_length > 0:
            box = gmsh.model.occ.addBox(index[0], index[1], index[2], 1, 1, box_length)
            gmsh_boxes.append(box)
    channel = gmsh.model.occ.cut([(3, channel)], [(3, box) for box in gmsh_boxes])
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=3)
    marker = 11
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], marker)
    gmsh.model.setPhysicalName(volumes[0][0], marker, "conductor")
    surfaces = gmsh.model.occ.getEntities(dim=2)
    wall_marker = 15
    walls = []
    for surface in surfaces:
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        if np.isclose(com[2], 0) or np.isclose(com[2], Lz) or np.isclose(com[1], 0) or np.isclose(com[1], Ly) or np.isclose(com[0], 0) or np.isclose(com[0], Lx):
            walls.append(surface[1])
    gmsh.model.addPhysicalGroup(2, walls, wall_marker)
    gmsh.model.setPhysicalName(2, wall_marker, "Walls")

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "FacesList", walls)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", resolution)
    gmsh.model.mesh.field.setNumber(2, "LcMax", 20 * resolution)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 1)

    gmsh.model.mesh.field.add("Min", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    
    gmsh.write(output_mshfile)
    
    return


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
    occlusions = np.logical_not(data)
    rectangles = make_rectangles(occlusions)
    boxes = make_boxes(rectangles)
    output_mshfile = f"mesh/s{grid_info}o{origin_str}_porous.msh"
    gmsh.initialize()
    build_voxels_mesh(boxes, output_mshfile)
    gmsh.finalize()
    print("writing xmdf tetrahedral mesh..")
    msh = meshio.read(output_mshfile)
    tetra_mesh = geometry.create_mesh(msh, "tetra")
    meshio.write(f"mesh/s{grid_info}o{origin_str}_tetr.xdmf", tetra_mesh)
