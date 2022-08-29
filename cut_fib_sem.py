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
sigma = 1.5

data = np.zeros((501, 501, 202))

for idx, img_file in enumerate(image_files):
    fname = str(idx).zfill(3) + ".tif"
    image = plt.imread(img_file)[1000-idx:1501-idx, 750:1251]
    img = np.zeros((501, 501))
    img[:, :] = image
    data[:, :, idx] = img

    img_raw = Image.fromarray(img.astype(np.uint8))
    # img_raw.save(os.path.join("unsegmented", fname), format="TIFF")
    img_laplace = filters.laplace(img)
    img_meijering = filters.meijering(img)
    img_sobel = filters.sobel(img)
    img_hysteresis = filters.apply_hysteresis_threshold(img, low=35, high=55)
    img_sato = filters.sato(img)
    img_gauss_diff = filters.difference_of_gaussians(img, 0.5, 2.5)
    img_segmented = np.digitize(img, bins=thresholds)
    fig, ax = plt.subplots(2, 4)
    ax[0, 0].imshow(img, cmap='gray')
    ax[0, 0].set_title("Original")

    ax[0, 1].imshow(img_laplace, cmap='gray')
    ax[0, 1].set_title("Laplace Filter")

    ax[0, 2].imshow(img_meijering, cmap='magma')
    ax[0, 2].set_title("Meijering Filter")

    ax[0, 3].imshow(img_sobel, cmap='magma')
    ax[0, 3].set_title("Sobel Filter")

    ax[1, 0].imshow(img_hysteresis, cmap='gray')
    ax[1, 0].set_title("Hysteresis Filter")

    ax[1, 1].imshow(img_sato, cmap='magma')
    ax[1, 1].set_title("Sato Filter")

    ax[1, 2].imshow(img_gauss_diff, cmap='gray')
    ax[1, 2].set_title("Difference of Gaussians Filter")

    img_gauss_diff = filters.meijering(img_gauss_diff)

    thresholds2 = filters.threshold_multiotsu(img_gauss_diff)
    segmented = np.digitize(img_gauss_diff, bins=thresholds2)
    merged_segmented = segmented == 0
    # ax[1, 3].imshow(merged_segmented, cmap='gray')
    # ax[1, 3].set_title("Segmented After Meijering and Multiotsu Threshold Filters")
    data = np.zeros((*merged_segmented.shape, 1))
    data[:, :, 0] = merged_segmented == 0
    n_points = int(np.sum(data))
    points = connected_pieces.build_points(data)
    new_points = {}
    for k, v in points.items():
        x, y, _ = k
        new_points[(x, y)] = v
    points = new_points
    points_view = {v: k for k, v in points.items()}
    edges = set()
    counter = 0
    for k in points.keys():
        x, y = k
        neighbors = [
            (int(x + 1), y),
            (int(x - 1), y),
            (x, int(y + 1)),
            (x, int(y - 1)),
        ]
        p0 = points[k]
        for neighbor in neighbors:
            p = points.get(neighbor)
            if p is None:
                continue
            edges.add(tuple(sorted([p0, p])))
    incidence_matrix = np.zeros((n_points, len(edges)), dtype=np.bool_)
    edge_id = 0
    for edge in edges:
        p0, p1 = edge
        incidence_matrix[p0, edge_id] = 1
        incidence_matrix[p1, edge_id] = 1
        edge_id += 1
    ns = linalg.null_space(incidence_matrix)
    # ns = ns * np.sign(ns[0,0])
    # pieces = [piece for piece in pieces]
    # dummy_img = np.zeros(img.shape, dtype=np.uint8)
    # for i, x in enumerate(pieces):
    #     if len(x) <= 50:
    #         continue
    # for p in pieces[3]:
    #     x, y, _ = points_view[p]
    #     dummy_img[x, y] = 1
    # ax[1, 3].imshow(dummy_img, cmap="gray")
    # ax[1, 3].set_title("Example Loop")
    # # ax[1, 3].hist(img_meijering.ravel(), bins=255)
    # # ax[1, 3].set_title('Sato Filter Histogram')
    # plt.show()
    break