#! /usr/bin/env python3
import copy
import os
import time

import argparse
import logging
import networkx as nx
import numpy as np

from skimage import io

import configs, filter_voxels, utils


FORMAT = '%(asctime)s: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__file__)
logger.setLevel('INFO')


def build_points(data, dp=0):
    """
    key: (x,y,z) coordinate
    value: point_id
    """
    points = {}
    count = 0
    for idx, v in np.ndenumerate(data):
        if v == 1:
            coord = idx
            if dp > 0:
                coord = (round(coord[0], dp), round(coord[1], dp), round(coord[2], dp))
            points[coord] = count
            count += 1
    return points


def build_graph(points, h=1, dp=0):
    """"""
    G = nx.Graph()
    for v in points.values():
        G.add_node(v)
    for k in points.keys():
        x, y, z = k
        if dp == 0:
            neighbors = [
                (int(x + 1), y, z),
                (int(x - 1), y, z),
                (x, int(y + 1), z),
                (x, int(y - 1), z),
                (x, y, int(z + 1)),
                (x, y, int(z - 1)),
            ]
        else:
            neighbors = [
                (round(x + h, dp), y, z),
                (round(x - h, dp), y, z),
                (x, round(y + h, dp), z),
                (x, round(y - h, dp), z),
                (x, y, round(z + h, dp)),
                (x, y, round(z - h, dp)),
            ]
        p0 = points[k]
        for neighbor in neighbors:
            p = points.get(neighbor)
            if p is None:
                continue
            G.add_edge(p0, p)
    return G


def add_boundary_points(points, x_max=50, y_max=50, z_max=50, h=0.5, dp=1):
    """
    A thickness of *h* pixels around the points of one phase to ensure continuity between phases.
    """
    new_points = copy.deepcopy(points)
    max_id = max(new_points.values())
    for (x0, y0, z0), _ in points.items():
        for sign_x in [-1, 0, 1]:
            for sign_y in [-1, 0, 1]:
                for sign_z in [-1, 0, 1]:
                    coord = (round(x0 + h * sign_x, dp), round(y0 + h * sign_y, dp), round(z0 + h * sign_z, dp))
                    if coord[0] > x_max or coord[1] > y_max or coord[2] > z_max:
                        continue
                    if np.less(coord, 0).any():
                        continue
                    v = new_points.get(coord)
                    if v is None:
                        max_id += 1
                        new_points[coord] = max_id

    return new_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='computes specific area')
    parser.add_argument('--img_folder', help='bmp files directory',
                        required=True)
    parser.add_argument('--grid_info', help='Nx-Ny-Nz',
                        required=True)
    parser.add_argument('--origin', default=(0, 0, 0), help='where to extract grid from')
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

    points = build_points(voxels, dp=1)
    points = add_boundary_points(points, x_max=Lx, y_max=Ly, z_max=Lz, h=0.5, dp=1)
    points_view = {v: k for k, v in points.items()}

    G = build_graph(points)
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