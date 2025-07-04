#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

from PIL import Image

import plot_opts, utils

plt.rcParams.update(plot_opts.params)


INPUT_DATA_DIR = "OneDrive/PhD/Data/SEM Image"
OUTPUT_DATA_DIR = "output/segmentation"


def create_2d_slices(img_num, outdir):
    """"""
    img_file = os.path.join(os.environ["HOME"], INPUT_DATA_DIR, f"SEM Image - SliceImage - {str(img_num).zfill(3)}.tif")
    img = plt.imread(img_file)[975-img_num-1:1475-img_num-1, 775:1275]
    pil_img = Image.fromarray(img)
    output_file = os.path.join(outdir, f"{str(img_num).zfill(3)}.tif")
    pil_img.save(output_file)


if __name__ == '__main__':
    img_first = os.path.join(os.environ["HOME"], INPUT_DATA_DIR, "SEM Image - SliceImage - 001.tif")
    img_last = os.path.join(os.environ["HOME"], INPUT_DATA_DIR, "SEM Image - SliceImage - 202.tif")
    slices_dir = os.path.join(OUTPUT_DATA_DIR, "raw")
    utils.make_dir_if_missing(slices_dir)
    for idx in range(1, 203):
        create_2d_slices(idx, slices_dir)
    # nz = 201
    # img_0 = plt.imread(img_first)[975:1475, 775:1275]
    # img_1 = plt.imread(img_last)[975-nz:1475-nz, 775:1275]
    # nx, ny = img_0.shape
    
    # print(np.unique(ski.filters.meijering(img_0, mode="nearest", alpha=0.05)))
    # thres_a = 52.5
    # thres_b = 17.5
    # img_0_new = ski.filters.meijering(img_0, mode="nearest", alpha=0.05)>0.1
    # ski.morphology.remove_small_objects(img_0_new, 10, out=img_0_new)

    # img_1_new = ski.filters.meijering(img_1, mode="nearest", alpha=0.05)>0.1
    # ski.morphology.remove_small_objects(img_1_new, 10, out=img_1_new)
    # mask = ski.morphology.remove_small_holes(img_1_new, 50)#, ski.morphology.disk(3))

#     cv = ski.segmentation.chan_vese(
#     img_0_new,
#     mu=0.25,
#     lambda1=1,
#     lambda2=1,
#     tol=1e-3,
#     max_num_iter=200,
#     dt=0.05,
#     init_level_set="checkerboard",
#     extended_output=True,
# )
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(mask, cmap='gray')
    # ax[0].grid()
    # ax[0].set_box_aspect(1)
    # ax[1].imshow(img_1_new, cmap='gray')
    # ax[1].grid()
    # ax[1].set_box_aspect(1)
    # plt.tight_layout()
    # plt.show()
    # fig, ax = plt.subplots()
    # ax.hist(img_1)
    # plt.show()
