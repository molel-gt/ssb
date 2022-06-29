#!/usr/bin/env python3
# coding: utf-8

import sys

import h5py
import numpy as np


current_filename = sys.argv[1]
Ly = float(sys.argv[2])
data = h5py.File(current_filename, "r")
geom = data['/Mesh/Grid/geometry']
values = data['/Function/f/0']

current_0 = []
current_1 = []

for idx, v in enumerate(values):
    _, y, _ = geom[idx]
    if len(v) == 3:
        v0 = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
    else:
        v0 = v
    if np.isclose(y, 0):
        current_0.append(v0)
    elif np.isclose(y, Ly):
        current_1.append(v0)
i_avg_1 = np.around(np.nanmean(current_0), 6)
i_avg_2 = np.around(np.nanmean(current_1), 6)
print("%0.2f,%0.6f,%0.6f" % (np.around(float(current_filename.split("-")[-2]), 2), i_avg_1, i_avg_2))