#! /usr/bin/env python3

import os
import time

import argparse
import logging
import networkx as nx
import numpy as np

import geometry, utils


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


def build_graph(points):
    """"""
    G = nx.Graph()
    for v in points.values():
        G.add_node(v)
    for k in points.keys():
        x, y, z = k
        neighbors = [
            (int(x + 1), y, z),
            (int(x - 1), y, z),
            (x, int(y + 1), z),
            (x, int(y - 1), z),
            (x, y, int(z + 1)),
            (x, y, int(z - 1)),
        ]
        p0 = points[k]
        for neighbor in neighbors:
            p = points.get(neighbor)
            if p is None:
                continue
            G.add_edge(p0, p)
    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='computes specific area')
    parser.add_argument('--img_folder', help='bmp files directory',
                        required=True)
    parser.add_argument('--grid_info', help='Nx-Ny-Nz',
                        required=True)
    parser.add_argument('--origin', default=(0, 0, 0), help='where to extract grid from')
    parser.add_argument("--resolution", nargs='?', const=1, default=0.5, type=float)

    args = parser.parse_args()
    if isinstance(args.origin, str):
        origin = tuple(map(lambda v: int(v), args.origin.split(",")))
    else:
        origin = args.origin
    origin_str = "-".join([str(v).zfill(3) for v in origin])
    grid_info = "-".join([v.zfill(3) for v in args.grid_info.split("-")])
    grid_size = int(args.grid_info.split("-")[0])
    Nx, Ny, Nz = [int(v) for v in args.grid_info.split("-")]
    img_dir = args.img_folder
    utils.make_dir_if_missing(f"mesh/{grid_info}_{origin_str}")
    im_files = sorted([os.path.join(img_dir, f) for
                       f in os.listdir(img_dir) if f.endswith(".bmp")])
    n_files = len(im_files)

    start_time = time.time()

    data = geometry.load_images_to_voxel(im_files, x_lims=(0, Nx),
                                         y_lims=(0, Ny), z_lims=(0, Nz), origin=origin)
    num_points = np.sum(data)
    points = build_points(data)
    points_view = {v: k for k, v in points.items()}
    G = build_graph(points)
    pieces = nx.connected_components(G)
    pieces = [piece for piece in pieces]
    logger.info("{:,} components".format(len(pieces)))
    for idx, piece in enumerate(pieces):
        largest_piece = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
        for p in piece:
            coord = points_view[p]
            largest_piece[coord] = 1
        if np.all(largest_piece[:, 0, :] == 0) or np.all(largest_piece[:, Ny - 1, :] == 0):
            logger.debug(f"Piece {idx} does not span both ends")
            continue
        logger.info(f"Piece {idx} spans both ends along y-axis")
        # occlusions = np.logical_not(largest_piece)
        # rectangles = mesher.make_rectangles(occlusions)
        # boxes = mesher.make_boxes(rectangles)
        # logger.info("No. voxels       : %s" % np.sum(occlusions))
        # logger.info("No. rectangles   : %s" % np.sum(rectangles))
        # logger.info("No. boxes        : %s" % np.sum(boxes))
        # output_mshfile = f"mesh/{grid_info}_{origin_str}/{idx}.msh"
        # gmsh.initialize()
        # gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0.5)
        # gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
        # gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0.5)  # FIXME
        # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", args.resolution)
        # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
        # gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        # gmsh.option.setNumber("Mesh.Smoothing", 500)
        # gmsh.option.setNumber("Mesh.AllowSwapAngle", 90)
        # mesher.build_voxels_mesh(boxes, output_mshfile)
        # gmsh.finalize()
        # logger.info("writing xmdf tetrahedral mesh..")
        # msh = meshio.read(output_mshfile)
        # tetra_mesh = geometry.create_mesh(msh, "tetra")
        # meshio.write(f"mesh/{grid_info}_{origin_str}/{idx}tetr.xdmf", tetra_mesh)
        # tria_mesh = geometry.create_mesh(msh, "triangle")
        # meshio.write(f"mesh/{grid_info}_{origin_str}/{idx}tria.xdmf", tria_mesh)
        # logger.info("Operation took {:,} seconds".format(int(time.time() - start_time)))
