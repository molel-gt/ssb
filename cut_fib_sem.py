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

img_dir = "SEM Image/"
image_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".tif")])

thresholds = [35, 65]
sigma = 1.5

data = np.zeros((501, 501, 202))

for idx, img_file in enumerate(image_files):
    fname = str(idx).zfill(3) + ".tif"
    image = plt.imread(img_file)[1000-idx:1501-idx, 750:1251]
    img = np.zeros((501, 501))
    img[:, :] = image
    data[:, :, idx] = img

    img_raw = Image.fromarray(img.astype(np.uint8))
    img_raw.save(os.path.join("unsegmented", fname), format="TIFF")
