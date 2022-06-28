#!/usr/bin/env python3
# coding: utf-8

import sys

import h5py
import numpy as np


current_filename = sys.argv[1]
data = h5py.File(current_filename, "r")
geom = data['/Mesh/Grid/geometry']
values = data['/Function/f/0']

current_0 = []
current_1 = []
current_2 = []

for idx, v in enumerate(values):
    x, y, z = geom[idx]
    if np.isclose(y, 0):
        current_0.append(v)
    elif np.isclose(y, 1):
        current_1.append(v)
    elif np.isclose(y, 2):
        current_2.append(v)
print("@ y = 0, current is", np.around(np.nanmean(current_0), 4))
print("@ y = 1, current is", np.around(np.average(current_1), 4))
print("@ y = 2, current is", np.around(np.average(current_2), 4))