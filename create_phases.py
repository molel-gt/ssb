#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import numpy as np
from tifffile import imsave
from PIL import Image

input_dir = os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/Archive")
output_dir = os.path.join(os.environ["WORK_DIR"], "output/segmented")


if __name__ == "__main__":
    for idx in range(3, 206):
        img = plt.imread(os.path.join(input_dir, f"{idx}.tif46.tif116.tif105.tif102.tif"))[:, :, 2]
        print(img.shape)
        cam_coords = np.isclose(img, 0)
        sse_coords = img > 130
        img_out = np.ones(img.shape, dtype=np.uint8)
        img_out[cam_coords] = 3
        img_out[sse_coords] = 2
        imsave(os.path.join(output_dir, f"{idx-2}.tif"), img_out[:, :])

