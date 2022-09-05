#!/usr/bin/env python3
import itertools
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from PIL import Image
from scipy import linalg, sparse
from skimage import (
    filters,
    segmentation,
    )

import connected_pieces, utils

utils.make_dir_if_missing("unsegmented")

img_dir = "SEM Image/"
image_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".tif")])
# 35 to 65
thresholds = [40, 65]
thresholds2 = [47, 65]
thresholds2 = [35, 65]
sigma = 1

data = np.zeros((501, 501, 202))
all_thresholds = []

for img_file in image_files:
    idx = int(img_file.split("-")[-1].strip().split(".")[0]) - 1
    if idx != 201:
        continue
    fname = str(idx).zfill(3) + ".tif"
    image = plt.imread(img_file)

    Nx = 641
    Ny = 961
    img = np.zeros((Nx, Ny))
    # img[:, :] = image[1000-idx:1501-idx, 750:1251]
    # image2 = image[360:1001, 560:1521]
    image2 = image[340:981, 740-idx:1701-idx]
    img = image2

    img_raw = Image.fromarray(img.astype(np.uint8))
    img_raw.save(os.path.join("unsegmented", fname), format="TIFF")
    img_gauss = filters.gaussian(img)
    thresholds_auto = filters.threshold_multiotsu(img_gauss)
    all_thresholds.append(thresholds_auto)
    img_segmented = filters.meijering(img_gauss)

    # img_out = Image.fromarray(img_segmented.astype(np.uint8))
    # img_out.save(os.path.join("segmented", fname), format="TIFF")

    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original")

    ax[1].imshow(img, cmap="gray")
    ax[1].set_title('Selected Section')

    ax[2].imshow(img_gauss, cmap='gray')
    ax[2].set_title("Gaussian Blur")

    ax[3].imshow(img_segmented, cmap='gray')
    ax[3].set_title("Edges Post Gaussian Blur")

    plt.show()
    break
all_thresholds = np.array(all_thresholds)
print(np.average(all_thresholds[:, 0]), np.average(all_thresholds[:, 1]))