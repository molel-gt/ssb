#!/usr/bin/env python3
# coding: utf-8

import logging
import os
import time
import warnings
warnings.filterwarnings("ignore")

import argparse
import gmsh
import meshio
import numpy as np

from itertools import groupby
from operator import itemgetter

import connected_pieces, geometry, utils

FORMAT = '%(asctime)s: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__file__)
logger.setLevel('INFO')


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

                if (voxels[p0] and voxels[p1] and voxels[p2] and voxels[p3]):
                    rectangles[p0] = 1

    return rectangles


def make_boxes(rectangles):
    Nx, Ny, _ = rectangles.shape
    boxes = np.zeros(rectangles.shape, dtype=np.uint16)
    for idx in range(Nx - 1):
        for idy in range(Ny - 1):
            box_length = 0
            start_pos = 0
            pieces = []
            rect_positions = [v[0] for v in np.argwhere(rectangles[idx, idy, :] != 0 )]
            if len(rect_positions) < 2:
                continue
            for _, g in groupby(enumerate(rect_positions), lambda ix: ix[0] - ix[1]):
                pieces.append(list(map(itemgetter(1), g)))
            for p in pieces:
                box_length = p[len(p) - 1] - p[0]
                start_pos = p[0]
                boxes[idx, idy, start_pos] = box_length
    return boxes


def build_voxels_mesh(output_mshfile, voxels, points_view={}):
    gmsh.model.add("3D")
    Nx, Ny, Nz = voxels.shape
    Lx = Nx - 1
    Ly = Ny - 1
    Lz = Nz - 1
    counter = 1
    gmsh_boxes = []
    channel = [(3, gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz))]

    logger.info("Adding volumes..")
    for idx in range(Nx):
        for idy in range(Ny):
            for idz in range(Nz):
                points = [
                    (idx, idy, idz), (idx + 1, idy, idz), (idx, idy + 1, idz), (idx, idy, idz + 1),
                    (idx + 1, idy + 1, idz), (idx + 1, idy, idz + 1), (idx, idy + 1, idz + 1), (idx + 1, idy + 1, idz + 1),
                ]
                makes_box = True
                for p in points:
                    if points_view.get(p) is None:
                        makes_box = False
                if makes_box:
                    box = (3, gmsh.model.occ.addBox(idx, idy, idz, 1, 1, 1))
                    gmsh_boxes.append(box)
    logger.info("Cutting occlusions..")
    gmsh.model.occ.cut(channel, gmsh_boxes)
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=3)
    logger.info("Setting physical groups..")

    for i, volume in enumerate(volumes):
        marker = int(counter + i)
        gmsh.model.addPhysicalGroup(volume[0], [volume[1]], marker)
        gmsh.model.setPhysicalName(volume[0], marker, f"V{marker}")
    logger.info("Set physical groups.")
    surfaces = gmsh.model.occ.getEntities(dim=2)
    insulated = []
    left_cc = []
    right_cc = []
    for surface in surfaces:
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        if np.isclose(com[1], 0):
            left_cc.append(surface[1])
        elif np.isclose(com[1], Ly):
            right_cc.append(surface[1])
        else:
            insulated.append(surface[1])
    y0_tag = gmsh.model.addPhysicalGroup(2, left_cc)
    gmsh.model.setPhysicalName(2, y0_tag, "left_cc")
    yl_tag = gmsh.model.addPhysicalGroup(2, right_cc)
    gmsh.model.setPhysicalName(2, yl_tag, "right_cc")
    insulated = gmsh.model.addPhysicalGroup(2, insulated)
    gmsh.model.setPhysicalName(2, insulated, "insulated")
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
    parser.add_argument("--resolution", nargs='?', const=1, default=0.5, type=float)
    parser.add_argument("--phase", nargs='?', const=1, default="electrolyte")

    args = parser.parse_args()
    phase = args.phase
    start = time.time()
    if isinstance(args.origin, str):
        origin = tuple(map(lambda v: int(v), args.origin.split(",")))
    elif isinstance(args.origin, tuple):
        origin = args.origin
    origin_str = "-".join([str(v).zfill(3) for v in origin])
    grid_info = "-".join([v.zfill(3) for v in args.grid_info.split("-")])
    grid_size = int(args.grid_info.split("-")[0])
    Nx, Ny, Nz = [int(v) for v in args.grid_info.split("-")]
    img_dir = args.img_folder
    utils.make_dir_if_missing(f"mesh/{phase}/{grid_info}_{origin_str}")
    im_files = sorted([os.path.join(img_dir, f) for
                       f in os.listdir(img_dir) if f.endswith(".bmp")])
    n_files = len(im_files)

    voxels = geometry.load_images_to_voxel(im_files, x_lims=(0, Nx),
                                         y_lims=(0, Ny), z_lims=(0, Nz), origin=origin)
    occlusions = np.logical_not(voxels)
    # rectangles = make_rectangles(occlusions)
    # boxes = make_boxes(rectangles)
    # logger.info("No. voxels       : %s" % np.sum(occlusions))
    # logger.info("No. rectangles   : %s" % np.sum(rectangles))
    # logger.info("No. boxes        : %s" % np.sum(boxes))
    points = connected_pieces.build_points(occlusions)
    points_view = {v: k for k, v in points.items()}
    output_mshfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous.msh"
    gmsh.initialize()
    # gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    # gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    # gmsh.option.setNumber("Mesh.Smoothing", 500)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", args.resolution)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
    # gmsh.option.setNumber("Mesh.AllowSwapAngle", 90)
    build_voxels_mesh(output_mshfile, occlusions, points_view)
    gmsh.finalize()
    logger.info("writing xmdf tetrahedral mesh..")
    msh = meshio.read(output_mshfile)
    tetra_mesh = geometry.create_mesh(msh, "tetra")
    meshio.write(f"mesh/{phase}/{grid_info}_{origin_str}/tetr.xdmf", tetra_mesh)
    tria_mesh = geometry.create_mesh(msh, "triangle")
    meshio.write(f"mesh/{phase}/{grid_info}_{origin_str}/tria.xdmf", tria_mesh)
    stop = time.time()
    logger.info("Took {:,}s".format(int(stop - start)))
