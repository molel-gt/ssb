#!/usr/bin/env python3
# coding: utf-8

import csv
import os

import argparse
import h5py
import logging
import networkx as nx
import numpy as np

import geometry, connected_pieces


FORMAT = '%(asctime)s: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel('INFO')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='computes specific area')
    parser.add_argument('--img_folder', help='bmp files directory',
                        required=True)
    args = parser.parse_args()
    img_dir = args.img_folder
    current_data_files = [os.path.join("output", f) for f in os.listdir("output") if f.endswith("current.h5")]
    with open("results.csv", "w") as fp:
        writer = csv.DictWriter(fp, fieldnames=["filename", "porosity (at 0)", "porosity (at L)", "kappa_eff (at 0)", "kappa_eff (at L)", "porosity (avg)", "kappa_eff (avg)", "bruggeman", "deviation @cc"])
        writer.writeheader()
        for fname in current_data_files:
            grid_info = fname.split("/")[-1].split("o")[0].strip("s")
            Nx, Ny, Nz = [int(v) for v in grid_info.split("-")]
            Lx = int(Nx - 1)
            Ly = int(Ny - 1)
            Lz = int(Nz - 1)
            origin_str = '_'.join(fname.split("o")[-1].split("_current")[0].split(","))
            origin = [int(v) for v in origin_str.split("_")]
            im_files = sorted([os.path.join(img_dir, f) for
                            f in os.listdir(img_dir) if f.endswith(".bmp")])
            n_files = len(im_files)

            voxels = geometry.load_images_to_voxel(im_files, x_lims=(0, Nx),
                                                y_lims=(0, Ny), z_lims=(0, Nz), origin=origin)
            num_points = np.sum(voxels)
            points = connected_pieces.build_points(voxels)
            points_view = {v: k for k, v in points.items()}
            G = connected_pieces.build_graph(points)
            pieces = nx.connected_components(G)
            pieces = [piece for piece in pieces]
            logger.info("{:,} components".format(len(pieces)))
            connected = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
            for idx, piece in enumerate(pieces):
                largest_piece = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
                for p in piece:
                    coord = points_view[p]
                    largest_piece[coord] = 1
                if np.all(largest_piece[:, 0, :] == 0) or np.all(largest_piece[:, Ny - 1, :] == 0):
                    logger.debug(f"Piece {idx} does not span both ends")
                    continue
                logger.info(f"Piece {idx} spans both ends along y-axis")
                connected += largest_piece
            voxels = connected
            eps_left = np.around(np.average(voxels[:, 0, :]), 4)
            eps_right = np.around(np.average(voxels[:, Ly, :]), 4)
            eps = np.around(np.average(voxels), 4)

            resultsdata = h5py.File(fname, "r")
            values = resultsdata['/Function/f/0']
            geom = resultsdata['/Mesh/Grid/geometry']
            vals_left = []
            vals_right = []
            for idx, (vx, vy, vz) in enumerate(values):
                coord = geom[idx, :]
                if np.isclose(coord[1], 0):
                    mag = (vx ** 2 + vy ** 2 + vz ** 2) ** (0.5)
                    vals_left.append(mag)
                if np.isclose(coord[1], Ly):
                    mag = (vx ** 2 + vy ** 2 + vz ** 2) ** (0.5)
                    vals_right.append(mag)

            # area-averaged local current
            current_left = np.around(eps_left * np.average(vals_left), 4)
            current_right = np.around(eps_right * np.average(vals_right), 4)
            kappa_eff_left = np.around(eps_left * current_left * Ly, 4)
            kappa_eff_right = np.around(eps_right * current_right * Ly, 4)
            row = {"filename": fname, "porosity (at 0)": eps_left, "porosity (at L)": eps_right,
                   "kappa_eff (at 0)": kappa_eff_left, "kappa_eff (at L)": kappa_eff_right,
                   "porosity (avg)": eps, "kappa_eff (avg)": np.around(0.5 * (kappa_eff_left + kappa_eff_right), 4), "bruggeman": np.around(eps ** 1.5, 4),
                   "deviation @cc": '-' if kappa_eff_left == 0 else 100 * abs(kappa_eff_right - kappa_eff_left) / (0.5 * (kappa_eff_right + kappa_eff_left)),
                   }
            writer.writerow(row)