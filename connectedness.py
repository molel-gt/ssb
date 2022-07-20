#!/usr/bin/env python3
import random
import os

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import pyvista as pv
from sklearn import neighbors

from stl import mesh

import geometry, mesher, utils


Nx = 203
Ny = 451
Nz = 801
img_dir = "/home/emolel3/dev/ssb/Archive/electrolyte"
im_files = sorted([os.path.join(img_dir, f) for
                       f in os.listdir(img_dir) if f.endswith(".bmp")])
n_files = len(im_files)

voxels = geometry.load_images_to_voxel(im_files, x_lims=(0, Nx),
                                        y_lims=(0, Ny), z_lims=(0, Nz))
av_x = np.mean(voxels, axis=0)
av_z = np.mean(voxels, axis=2)
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.imshow(av_x)
ax1.set_title("Average along x")
ax2.imshow(av_z)
ax2.set_title("Average along z")
plt.show()