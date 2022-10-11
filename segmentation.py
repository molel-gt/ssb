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
    fname = str(idx).zfill(3) + ".tif"
    image = plt.imread(img_file)

    Nx = 501
    Ny = 501
    img = np.zeros((Nx, Ny))
    # img[:, :] = image[1000-idx:1501-idx, 750:1251]
    # image2 = image[360:1001, 560:1521]
    # image2 = image[340:981, 740:1701]
    image2 = image[1000-idx:1501-idx, 750:1251]
    img = image2

    img_raw = Image.fromarray(img.astype(np.uint8))
    img_raw.save(os.path.join("unsegmented", fname), format="TIFF")
    # img_gauss = filters.gaussian(img)
    # thresholds_auto = filters.threshold_multiotsu(img_gauss)
    # all_thresholds.append(thresholds_auto)
    # img_segmented = filters.meijering(img_gauss)
    # img_edges = img_segmented >= 0.075
    # test small slice due to memory issues
    # img_edges = img_edges[0:101, 0:101]
    # edges_coords = np.argwhere(img_edges == 1)
    # points = {}
    # for idx, coord in enumerate(edges_coords):
    #     points[tuple(coord)] = idx

    # n_points = int(np.max(list(points.values())) + 1)
    # edges = set()
    # for coord, origin_idx in points.items():
    #     x, y = coord
    #     neighbors = [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]
    #     for check_point in neighbors:
    #         neighbor_idx = points.get(check_point)
    #         if neighbor_idx is not None:
    #             edges.add(
    #                 tuple(sorted([origin_idx, neighbor_idx]))
    #             )
    # edges = list(edges)
    # n_edges = len(edges)
    # # adjacency matrix
    # A = np.zeros((n_edges, n_points), dtype=np.bool_)
    # new_edges = {}
    # for idx, edge in enumerate(edges):
    #     new_edges[edge] = idx
    #     p1, p2 = edge
    #     A[idx, p1] = 1
    #     A[idx, p2] = 1
    # print(A.shape)
    # ns = linalg.null_space(A.T)
    # ns = np.abs(ns)  # * np.sign(ns[0, 0])
    # print(np.min(ns[:, 0]), np.max(ns[:, 0]))
    # ns = np.logical_not(np.isclose(ns, 0))
   
    # for idx in range(ns.shape[1]):
    #     print(f"Loop {idx} has %d points" % np.sum(ns[:, idx]))
    # print("Number of loops: ", ns.shape[1])

    # img_out = Image.fromarray(img_segmented.astype(np.uint8))
    # img_out.save(os.path.join("segmented", fname), format="TIFF")

    # fig, ax = plt.subplots(1, 5)
    # ax[0].imshow(image, cmap='gray')
    # ax[0].set_title("Original")

    # ax[1].imshow(img, cmap="gray")
    # ax[1].set_title('Selected Section')

    # ax[2].imshow(img_gauss, cmap='gray')
    # ax[2].set_title("Gaussian Blur")

    # ax[3].imshow(img_edges, cmap='gray')
    # ax[3].set_title("Edges Post Gaussian Blur")

    # ax[4].hist(img.ravel(), density=True, bins=255)
    # ax[4].set_title("Image Histogram")

    # plt.show()
    # break
all_thresholds = np.array(all_thresholds)
print(np.average(all_thresholds[:, 0]), np.average(all_thresholds[:, 1]))