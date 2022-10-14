#! /usr/bin/env python3

import os
import time

import argparse
import logging
import networkx as nx
import numpy as np

from skimage import io

import configs, filter_voxels, geometry, utils


FORMAT = '%(asctime)s: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__file__)
logger.setLevel('INFO')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='computes specific area')
    parser.add_argument('--grid_info', help='Nx-Ny-Nz',
                        required=True)
    parser.add_argument("--phase", help='Phase that we want to reconstruct, e.g. 0 for void, 1 for solid electrolyte and 2 for active material', nargs='?', const=1, default=1, type=int)

    args = parser.parse_args()
    scaling = configs.get_configs()['VOXEL_SCALING']
    img_folder = configs.get_configs()['LOCAL_PATHS']['segmented_image_stack']
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])
    scale_factor = (scale_x, scale_y, scale_z)
    origin = [int(v) for v in configs.get_configs()['GEOMETRY']['origin'].split(",")]
    origin_str = "-".join([str(v).zfill(3) for v in origin])
    grid_info = args.grid_info
    grid_size = int(args.grid_info.split("-")[0])
    Nx, Ny, Nz = [int(v) for v in args.grid_info.split("-")]
    Lx = Nx - 1
    Ly = Ny - 1
    Lz = Nz - 1
    mesh_dir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], f"{args.phase}/{grid_info}_{origin_str}")
    utils.make_dir_if_missing(mesh_dir)

    im_files = sorted([os.path.join(img_folder, f) for
                       f in os.listdir(img_folder) if f.endswith(".tif")])
    n_files = len(im_files)

    start_time = time.time()

    shape = [*io.imread(im_files[0]).shape, n_files]
    voxels_raw = filter_voxels.load_images(im_files, shape)[origin[0]:Nx+origin[0], origin[1]:Ny+origin[1], origin[2]:Nz+origin[2]]
    voxels_filtered = filter_voxels.get_filtered_voxels(voxels_raw)
    voxels = np.isclose(voxels_filtered, args.phase)

    points = geometry.build_points(voxels, dp=1)
    points = geometry.add_boundary_points(points, x_max=Lx, y_max=Ly, z_max=Lz, h=0.5, dp=1)
    points_view = {v: k for k, v in points.items()}

    G = geometry.build_graph(points)
    pieces = nx.connected_components(G)
    pieces = [piece for piece in pieces]
    logger.info("{:,} components".format(len(pieces)))
    for idx, piece in enumerate(pieces):
        working_piece = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
        for p in piece:
            coord = points_view[p]
            working_piece[coord] = 1
        if np.all(working_piece[:, 0, :] == 0) or np.all(working_piece[:, Ly, :] == 0):
            logger.debug(f"Piece {idx} does not span both ends")
            continue
        logger.info(f"Piece {idx} spans both ends along y-axis")