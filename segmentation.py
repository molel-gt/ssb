#!/usr/bin/env python3
import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from skimage import (
    filters,
    segmentation,
    )

import utils

utils.make_dir_if_missing("unsegmented")
utils.make_dir_if_missing("segmented/activematerial")
utils.make_dir_if_missing("segmented/electrolyte")
utils.make_dir_if_missing("segmented/void")
img_dir = "SEM Image/"
image_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".tif")])

thresholds = [30, 80]

for idx, img_file in enumerate(image_files):
    fname = str(idx).zfill(3) + ".bmp"
    image = plt.imread(img_file)
    img = np.zeros((501, 501))
    img[:, :] = image[1000-idx:1501-idx, 750:1251]
    plt.imsave(os.path.join("unsegmented", str(idx).zfill(3) + ".bmp"), img)
    img = filters.gaussian(img, sigma=0.5, mode='wrap')
    # thresholds = filters.threshold_multiotsu(img)
    img = np.digitize(img, bins=thresholds)
    electrolyte_img = Image.fromarray(img == 2)
    activematerial_img = Image.fromarray(img == 1)
    void_img = Image.fromarray(img == 0)
    activematerial_img.save(f"segmented/activematerial/SegIm{fname}", format="bmp")
    electrolyte_img.save(f"segmented/electrolyte/SegIm{fname}", format="bmp")
    void_img.save(f"segmented/void/SegIm{fname}", format="bmp")

    # Using the threshold values, we generate the three regions.
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

    # Plotting the original image.
    ax[0].imshow(image[1000-idx:1501-idx, 750:1251], cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    # Plotting the Multi Otsu result.
    ax[1].imshow(img, cmap='jet')
    ax[1].set_title('Multi-Otsu result')
    ax[1].axis('off')

    # Plotting the histogram and the two thresholds obtained from
    # multi-Otsu.
    ax[2].hist(image[1000-idx:1501-idx, 750:1251].ravel(), bins=255)
    ax[2].set_title('Histogram')
    for thresh in thresholds:
        ax[2].axvline(thresh, color='r')

    plt.subplots_adjust()
    plt.show()