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


class Box:
    def __init__(self, origin, dx, dy, dz):
        x, y, z = origin
        self.origin = (x, y, z)
        self.dx = dx
        self.dy = dy
        self.dz = dz


def create_boxes(voxels, points_view):
    dummy_boxes = np.zeros(voxels.shape, dtype=np.uint8)
    boxes = np.zeros(voxels.shape, dtype=np.uint8)

    for coord, v in np.ndenumerate(voxels):
        x0, y0, z0 = coord
        makes_box = True
        cube_points = [
            (x0, y0, z0), (x0 + 1, y0, z0), (x0, y0 + 1, z0), (x0, y0, z0 + 1),
            (x0 + 1, y0 + 1, z0), (x0 + 1, y0, z0 + 1), (x0, y0 + 1, z0 + 1), (x0 + 1, y0 + 1, z0 + 1),
        ]
        for p in cube_points:
            if points_view.get(p) is None:
                makes_box = False
        if makes_box:
            dummy_boxes[coord] = 1

    for idx in range(Nx - 1):
        for idy in range(Ny - 1):
            l_box = 0
            start_pos = 0
            pieces = []
            rect_positions = [v[0] for v in np.argwhere(dummy_boxes[idx, idy, :] != 0 )]
            if len(rect_positions) < 1:
                continue
            for _, g in groupby(enumerate(rect_positions), lambda ix: ix[0] - ix[1] + 1):
                pieces.append(list(map(itemgetter(1), g)))
            for p in pieces:
                l_box = p[-1] - p[0] + 1
                if l_box <= 0:
                    continue
                start_pos = p[0]
                boxes[idx, idy, start_pos] = l_box
    return boxes


def build_voxels_mesh(output_mshfile, voxels=None, points_view=None):
    gmsh.model.add("3D")
    Nx, Ny, Nz = voxels.shape
    Lx = Nx - 1
    Ly = Ny - 1
    Lz = Nz - 1
    counter = 1
    gmsh_boxes = []
    channel = [(3, gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz))]
    boxes = np.zeros(voxels.shape, dtype=np.uint8)
    boxes = create_boxes(voxels, points)
    other_points = connected_pieces.build_points(np.logical_not(voxels))
    other_boxes = create_boxes(np.logical_not(voxels), other_points)

    for idx in range(Nx):
        for idy in range(Ny):
            for idz in range(Nz):
                box_length = boxes[idx, idy, idz]
                if box_length > 0:
                    counter += 1
                    box = gmsh.model.occ.addBox(idx, idy, idz, 1, 1, box_length)
                    gmsh_boxes.append((3, box))
    logger.info("Lower limit porosity : {:.4f}".format(np.average(other_boxes)))
    logger.info("Upper limit porosity : {:.4f}".format(1 - np.average(boxes)))
    logger.debug("Cutting out {:,} occlusions corresponding to insulator vol {:,}..".format(len(gmsh_boxes), np.sum(boxes)))
    gmsh.model.occ.cut(channel, gmsh_boxes)
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=3)
    logger.debug("Setting physical groups..")

    for i, volume in enumerate(volumes):
        marker = int(counter + i)
        vol_tag = gmsh.model.addPhysicalGroup(volume[0], [volume[1]])
        gmsh.model.setPhysicalName(3, vol_tag, f"VOLUME-{marker}")
    logger.debug("Done setting physical groups.")
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
    img_dir = os.path.join(args.img_folder, f'{phase}')
    utils.make_dir_if_missing(f"mesh/{phase}/{grid_info}_{origin_str}")
    old_im_files = sorted([os.path.join(img_dir, f) for
                       f in os.listdir(img_dir) if f.endswith(".bmp")])
    im_files = [""] * len(old_im_files)
    for i, f in enumerate(old_im_files):
        fdir = os.path.dirname(f)
        idx = int(f.split("/")[-1].split(".")[0].strip("SegIm"))
        im_files[idx - 3] = f
    n_files = len(im_files)

    voxels = geometry.load_images_to_voxel(im_files, x_lims=(0, Nx),
                                         y_lims=(0, Ny), z_lims=(0, Nz), origin=origin)
    eps = np.average(voxels)
    logger.info(f"Rough porosity       : {eps:.4f}")
    occlusions = np.logical_not(voxels)
    points = connected_pieces.build_points(occlusions)
    points_view = {v: k for k, v in points.items()}
    output_mshfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous.msh"
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", args.resolution)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
   
    build_voxels_mesh(output_mshfile, occlusions, points)
    gmsh.finalize()
    logger.debug("Writing xmdf tetrahedral mesh..")
    msh = meshio.read(output_mshfile)
    tetra_mesh = geometry.create_mesh(msh, "tetra")
    meshio.write(f"mesh/{phase}/{grid_info}_{origin_str}/tetr.xdmf", tetra_mesh)
    tria_mesh = geometry.create_mesh(msh, "triangle")
    meshio.write(f"mesh/{phase}/{grid_info}_{origin_str}/tria.xdmf", tria_mesh)
    stop = time.time()
    logger.info("Took {:,}s".format(int(stop - start)))
