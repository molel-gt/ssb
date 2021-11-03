#!/usr/bin/env python3

import os

import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='creates node file for meshing..')
    parser.add_argument('--results_dir', help='directory with .h5 and .xdmf files', required=True)
    parser.add_argument('--mesh_dir', help='directory with input mesh files', required=True)
    parser.add_argument('--results_file_name', help='results file name', required=True)
    parser.add_argument('--node_file_name', help='input node file name', required=True)
    args = parser.parse_args()

    input_file_path = os.path.join(args.results_dir, args.results_file_name + ".h5")

    # load simulation results
    with h5py.File(input_file_path, 'r') as hf:
        mesh_data = hf.get('Function').values()
        for obj_ in mesh_data:
            new_obj = obj_.values()
            for no_obj in new_obj:
                results = no_obj.value
    grid_size = int(args.mesh_dir.split('/')[-2])
    grid_data = np.empty([grid_size+1, grid_size+1, grid_size+1])

    # load geometry
    node_file_path = os.path.join(args.mesh_dir, args.node_file_name + ".node")
    with open(node_file_path, "r") as fp:
        reader = fp.readlines()
        for idx, row in enumerate(reader):
            if idx < 3:
                continue
            row_values = [int(float(v)) for v in row.strip("\n").split(" ")]
            coords = row_values[1:4]
            value = results[row_values[0]]
            grid_data[coords] = value
    [current, grady, gradz] = np.gradient(grid_data)
    new_current = np.sum(current, axis=(1, 2)) / (grid_size * grid_size)
    i_cell = np.average(new_current)
    delta_phi = 1 / grid_size
    print("conductivity: ", i_cell / delta_phi)
    plt.plot(np.linspace(0, grid_size, grid_size + 1), current[:, int(0.50 * grid_size), int(0.50 * grid_size)], 'bo')
    plt.plot(np.linspace(0, grid_size, grid_size + 1), new_current, 'r--')
    plt.grid()
    plt.show()
