#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import numpy as np
from tifffile import imwrite
from PIL import Image, ImageOps

import utils

input_dir = os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/Archive")
output_dir = os.path.join(os.environ["WORK_DIR"], "output/segmented")


if __name__ == "__main__":
    # for idx in range(1, 203):
    #     img = 20 * np.ones((500, 500), dtype=np.uint8)
    #     img_cam = plt.imread(f"output/segmentation/cam/{str(idx).zfill(3)}.tif")[:375, :]
    #     img_voids = plt.imread(f"output/segmentation/voids/{str(idx).zfill(3)}.tif")[:375, :]
    #     cam_coords = np.where(np.isclose(img_cam, 2))
    #     void_coords = np.where(img_voids > 0)
    #     img[cam_coords] = 30
    #     img[void_coords] = 10
    #     img[:13, :] = 30
    #     imwrite(os.path.join("output/segmentation/merged", f"{idx}.tif"), img[:, :])
    output_dir = os.path.join(os.environ["WORK_DIR"], "output/fib-sem-jg")
    utils.make_dir_if_missing(output_dir)
    for idx in range(3, 206):
        img = plt.imread(os.path.join(input_dir, f"{idx}.tif46.tif116.tif105.tif102.tif")).copy()
        print(img.shape)
        img = img[:375, :500, 2]
        print(img.shape)
        cam_coords = np.where(np.isclose(img, 0))
        sse_coords = np.where(img > 130)
        nx, ny = img.shape
        shape = (500, ny)
        img_out = 10 * np.ones(shape, dtype=np.uint8)

        img_out[cam_coords] = 30
        img_out[sse_coords] = 20
        # sse separator
        img_out[nx:, :] = 20
        # cam padding for contact with positive current collector
        img_out[:13, :] = 30
        # img_final = ImageOps.grayscale(Image.fromarray(img_out))
        # img_final.save(os.path.join(output_dir, f"{idx-2}.tif"))
        imwrite(os.path.join(output_dir, f"{idx-2}.tif"), img_out[:, :])

