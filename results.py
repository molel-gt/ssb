#!/usr/bin/env python3

import os

import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np

import create_node_files
import utils


def conductivity_eff(grid_id, input_file_path, node_file_path):
    """"""
    print("Processing `{}`".format(input_file_path))
    grid_size = int(grid_id.split('_')[0])
    grid_data = np.empty([grid_size+1, grid_size+1, grid_size+1])
    grid_data[:] = np.nan

    # load simulation results
    with h5py.File(input_file_path, 'r') as hf:
        values = np.asarray(hf.get("Function").get("f_4").get("0"))
        positions = np.asarray(hf.get("Mesh").get("Grid").get("geometry"))
    results = np.hstack((positions, values))
    for _, row in enumerate(results):
        coords = [int(v) for v in row[:3]]
        i_x, i_y, i_z = coords
        value = row[3]
        grid_data[i_x, i_y, i_z] = value

    plt.subplot(2, 1, 1)
    plt.title("Grid size: {}".format(grid_size))
    plt.scatter(np.linspace(0, grid_size, grid_size+1), grid_data[:, 34, 34])
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid()

    [u_x, _, _] = np.gradient(grid_data)
    u_x = np.clip(u_x, -100, 0)
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(0, grid_size, grid_size+1), u_x[:, 34, 34])
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("u_x(x)")
    plt.show()

    delta_phi_dx = -1 / grid_size
    kappa_eff = np.average(u_x) / delta_phi_dx

    return kappa_eff, u_x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='creates node file for meshing..')
    parser.add_argument('--results_dir', help='path to results folder for each grid',
                        required=True)
    parser.add_argument('--mesh_dir', help='path to input mesh files folder for each grid',
                        required=True)
    parser.add_argument('--input_bmp_dir', help='path to input bmp files',
                        required=True)
    parser.add_argument('--grid_sizes', help='description of grid size_startId_stopId',
                        type=lambda s: [int(item) for item in s.split(',')], required=True)
    args = parser.parse_args()
    n_files = 90
    size_delta = 5
    results_dir = args.results_dir
    mesh_dir = args.mesh_dir
    grid_sizes = args.grid_sizes
    summary = {}
    files_list = [os.path.join(args.input_bmp_dir, f) for f in os.listdir(args.input_bmp_dir) if f.endswith(".bmp")]
    grid_extents = utils.get_samples_of_test_grids(grid_sizes, n_files, size_delta)
    for i, grid_info in enumerate(grid_extents):
        (grid_size, start_pos, end_pos) = [int(v) for v in grid_info]
        grid_id = "_".join(map(str, [int(grid_size), int(start_pos), int(end_pos)]))
        results_sub_dir = os.path.join(results_dir, grid_id)
        input_file_path = os.path.join(results_sub_dir, "output.h5")
        node_file_path = os.path.join(mesh_dir, grid_id, "porous-solid.node")

        if not (os.path.exists(input_file_path) and os.path.exists(node_file_path)):
            continue

        sub_img_data = create_node_files.load_images_to_logical_array(files_list, [int(v) for v in grid_info])
        porosity = np.average(sub_img_data)
        kappa_eff, current = conductivity_eff(grid_id, input_file_path, node_file_path)
        summary[grid_id] = (porosity, kappa_eff)
    data = np.array(list([item for item in summary.values()]))

    plt.scatter(data[:, 0], data[:, 1])
    plt.grid()
    plt.show()
