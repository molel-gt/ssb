#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

import configs, utils


def H_lo(D, D0, n):
    return 1 / (1 + (D/D0) ** (2 * n))


def H_hi(D, D0, n):
    return 1 / (1 + (D0/D) ** (2 * n))


def fft_denoise(img, band_type='lo'):
    """"""
    if band_type not in ('lo', 'hi'):
        raise ValueError("Unknown band type")
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    M, N = img.shape
    H = np.zeros((M, N), dtype=np.float32)
    n = 1
    D0 = 0.75

    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M/2) ** 2 + (v - N/2) ** 2)
            if band_type == 'hi':
                H[u, v] = H_hi(D, D0, n)
            else:
                H[u, v] = H_lo(D, D0, n)

    Gshift = Fshift * H
    G = np.fft.ifftshift(Gshift)
    g = np.abs(np.fft.ifft2(G)) * 255
    
    return g


if __name__ == '__main__':
    unsegmented_dir = configs.get_configs()['LOCAL_PATHS']['unsegmented_image_stack']
    utils.make_dir_if_missing(unsegmented_dir)

    img_dir = "SEM Image/"
    image_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".tif")])

    for img_file in image_files:
        print(f"Processing image {img_file}")
        idx = int(img_file.split("-")[-1].strip().split(".")[0]) - 1
        fname = str(idx).zfill(3) + ".tif"
        image = plt.imread(img_file)

        Nx = 501
        Ny = 501
        img = np.zeros((Nx, Ny))
        image2 = image[1000-idx:1501-idx, 750:1251]
        img = image2

        g = fft_denoise(img, 'hi')

        img_raw = Image.fromarray(g.astype(np.uint8))
        img_raw.save(os.path.join(unsegmented_dir, fname), format="TIFF")
