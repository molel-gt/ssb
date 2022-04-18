#!/usr/bin/env python3

import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    current_file = sys.argv[1]
    data = h5py.File(current_file, "r")
    geom = data["/Mesh/Grid/geometry"]
    values = data["/Function/f/0"]
    vals_at_x0 = []
    vals_at_x1 = []

    pf = 1 - int(current_file.split('o')[-1].split("_")[0]) / 100

    for idx, coord in enumerate(geom):
        if np.isclose(coord[0], 0):
            vals_at_x0.append(values[idx])
        if np.isclose(coord[0], 1):
            vals_at_x1.append(values[idx])
    print("eff. conductivity @ x=0  :", np.average(vals_at_x0) * pf)
    print("eff. conductivity @ x=1  :", np.average(vals_at_x1) * pf)
    print("avg. eff. conductivity   :", np.average(values) * pf)
    print("bruggeman                :", pf ** 1.5)