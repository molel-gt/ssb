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

thresholds = [35, 65]
thresholds2 = [35, 85]
sigma = 0.5

data = np.zeros((501, 501, 202))

for idx, img_file in enumerate(image_files):
    fname = str(idx).zfill(3) + ".tif"
    image = plt.imread(img_file)[1000-idx:1501-idx, 750:1251]
    img = np.zeros((501, 501))
    img[:, :] = image
    data[:, :, idx] = img

    img_raw = Image.fromarray(img.astype(np.uint8))
    # img_raw.save(os.path.join("unsegmented", fname), format="TIFF")
    # img_laplace = filters.laplace(img)
    # img_meijering = filters.meijering(img)
    # img_sobel = filters.sobel(img)
    # img_hysteresis = filters.apply_hysteresis_threshold(img, low=35, high=55)
    # img_sato = filters.sato(img)
    img_gauss_diff = filters.difference_of_gaussians(img, 0.5, 2.5)
    img_gauss = filters.gaussian(img, sigma=sigma)
    img_segmented = np.digitize(img_gauss, bins=thresholds)
    img_segmented2 = np.digitize(img_gauss, bins=thresholds2)

    img_out = Image.fromarray(img_segmented.astype(np.uint8))
    img_out.save(os.path.join("segmented", fname), format="TIFF")

    # fig, ax = plt.subplots(1, 5)
    # ax[0].imshow(img, cmap='gray')
    # ax[0].set_title("Original")

    # ax[1].hist(img.ravel(), bins=255, density=True)
    # ax[1].set_title('Image Histogram')

    # ax[2].imshow(img_gauss_diff, cmap='gray')
    # ax[2].set_title("Gaussian Blur")

    # ax[3].imshow(img_segmented, cmap='gray')
    # ax[3].set_title(f"Segmented Gaussian Blur {thresholds.__repr__()}")

    # ax[4].imshow(img_segmented2, cmap='gray')
    # ax[4].set_title(f"Segmented Gaussian Blur {thresholds2.__repr__()}")

    # plt.show()
    # break