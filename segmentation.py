#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

import utils

utils.make_dir_if_missing("unsegmented")

img_dir = "SEM Image/"
image_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".tif")])

data = np.zeros((501, 501, 202))
all_thresholds = []

for img_file in image_files:
    idx = int(img_file.split("-")[-1].strip().split(".")[0]) - 1
    fname = str(idx).zfill(3) + ".tif"
    image = plt.imread(img_file)

    Nx = 501
    Ny = 501
    img = np.zeros((Nx, Ny))
    image2 = image[1000-idx:1501-idx, 750:1251]
    img = image2

    img_raw = Image.fromarray(img.astype(np.uint8))
    img_raw.save(os.path.join("unsegmented", fname), format="TIFF")
