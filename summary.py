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
def compute_resistance(coverage, w, h, voltages=(1, 2), Lx=100, Ly=1):
    currents = []
    for v in voltages:
        f = results_filename(coverage, w, h, v)
        data = h5py.File(f, "r")
        geom = data['/Mesh/Grid/geometry']
        values = data['/Function/f/0']
        current_0 = []
        current_1 = []

        for idx, v in enumerate(values):
            x, y, _ = geom[idx]
            if np.isclose(y, 0):
                if 0.5 * (1 - coverage) * Lx <= x <= (0.5 + 0.5 * coverage) * Lx:
                    current_0.append(v)
            elif np.isclose(y, Ly):
                current_1.append(v)
        i_avg_1 = np.around(np.nanmean(current_0), 6)
        i_avg_2 = np.around(np.nanmean(current_1), 6)
        currents.append((i_avg_1, i_avg_2))
    return (voltages[1] - voltages[0]) / (currents[1][1] - currents[0][1]), currents


def results_filename(coverage, w, h, voltage: int):
    return f'current_constriction/{h:.3}_{w:.3}_{coverage:.2}_{voltage}_current.h5'


if __name__ == '__main__':
    heights = [0.25, 0.50, 0.75]
    widths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    coverages = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    voltages = (1, 2)
    columns = ['coverage', 'voltage', 'Lx', 'Ly', 'slice_width', 'slice_y_position', 'current (y = 0)', 'current (y = Ly)', 'resistance']
    utils.make_dir_if_missing("current_constriction")
    with open("current_constriction/current-constriction.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for w in widths:
            for h in heights:
                for cov in coverages:
                    try:
                        resistance, currents = compute_resistance(cov, w, h, voltages=voltages)
                    except Exception as e:
                        print(e)
                        continue
                    print("processing coverage %0.2f, with slice width %d  at y = %0.2f" % (cov, int(w * 100), h))
                    writer.writerow(
                        {
                            'coverage': cov, 'voltage': "-", 'Lx': 100, 'Ly': 1, 'slice_width': w * 100,
                            'slice_y_position': h, 'current (y = 0)': "-", 'current (y = Ly)': "-",
                            'resistance': np.around(resistance, 2)
                        }
                    )
                    for idx, current in enumerate(currents):
                        writer.writerow(
                        {
                            'coverage': cov, 'voltage': voltages[idx], 'Lx': 100, 'Ly': 1, 'slice_width': w * 100,
                            'slice_y_position': h, 'current (y = 0)': currents[idx][0], 'current (y = Ly)': currents[idx][1],
                            'resistance': "-"
                        }
                    )