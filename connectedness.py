#!/usr/bin/env python3
import os

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

import geometry


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
av_xy = np.mean(av_x, axis=0)
av_zy = np.mean(av_z, axis=1)
fig, ((ax1, ax2), (ax11, ax22)) = plt.subplots(2, 2)
im1 = ax1.imshow(av_x)
ax1.set_title("Average along x")
im2 = ax2.imshow(av_z)
ax2.set_title("Average along z")

divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax1, orientation='vertical')

divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax2, orientation='vertical')

ax11.plot(av_xy)
ax11.set_title("Average along xy")
ax22.plot(av_zy)
ax22.set_title("Average along yz")
plt.show()