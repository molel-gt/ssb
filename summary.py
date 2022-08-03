#!/usr/bin/env python3
# coding: utf-8
import csv
import sys

import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np

import utils


# current constriction
def compute_resistance(coverage, w, h, voltages=(1, 2), Lx=100, Ly=1, pos='mid', n_pieces=0):
    currents = []
    for v in voltages:
        f = results_filename(coverage, w, h, v, pos, n_pieces)
        data = h5py.File(f, "r")
        geom = data['/Mesh/Grid/geometry']
        values = data['/Function/f/0']
        current_0 = []
        current_1 = []

        for idx, v in enumerate(values):
            x, y, _ = geom[idx]
            if np.isclose(y, 0):
                # if 0.5 * (1 - coverage) * Lx <= x <= (0.5 + 0.5 * coverage) * Lx:
                #     current_0.append(v)
                pass
            if np.isclose(y, Ly):
                current_1.append(v)
        i_avg_1 = np.nan # np.around(np.nanmean(current_0), 6)
        i_avg_2 = np.around(np.nanmean(current_1), 6)
        currents.append((i_avg_1, i_avg_2))
    return (voltages[1] - voltages[0]) / (currents[1][1] - currents[0][1]), currents


def results_filename(coverage, w, h, voltage: int, pos, n_pieces):
    return f'current_constriction/{h:.3}_{w:.3}_{coverage:.2}_{voltage}_pos-{pos}_pieces-{n_pieces}_current.h5'


if __name__ == '__main__':
    heights = [0.0]
    widths = [0.0]
    coverages = [0.25]
    voltages = (1, 2)
    n_pieces = [1, 2, 5, 10, 15, 50, 75, 100, 250, 350, 500]
    pos = ['mid']
    columns = ['coverage', 'voltage', 'Lx', 'Ly', 'slice_width', 'slice_y_position', 'current (y = 0)', 'current (y = Ly)', 'resistance', 'pos', 'n_pieces']
    utils.make_dir_if_missing("current_constriction")
    with open("current_constriction/current-constriction.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for w in widths:
            for h in heights:
                for cov in coverages:
                    for p in pos:
                        for num_piece in n_pieces:
                            try:
                                resistance, currents = compute_resistance(cov, w, h, voltages=voltages, pos=p, n_pieces=num_piece)
                            except Exception as e:
                                print(e)
                                continue
                            print("processing coverage %0.2f, with slice width %d  at y = %0.2f" % (cov, int(w * 100), h))
                            for idx, current in enumerate(currents):
                                writer.writerow(
                                {
                                    'coverage': cov, 'voltage': voltages[idx], 'Lx': 100, 'Ly': 1, 'slice_width': w * 100,
                                    'slice_y_position': h, 'current (y = 0)': currents[idx][0], 'current (y = Ly)': currents[idx][1],
                                    'resistance': "-", "pos": p, "n_pieces": num_piece,
                                })
                            writer.writerow(
                                {
                                    'coverage': cov, 'voltage': "-", 'Lx': 100, 'Ly': 1, 'slice_width': w * 100,
                                    'slice_y_position': h, 'current (y = 0)': "-", 'current (y = Ly)': "-",
                                    'resistance': np.around(resistance, 2), "pos": p, "n_pieces": num_piece,
                                }
                            )
