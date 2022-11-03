#! /usr/bin/env python3

import os
import subprocess

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from IPython.display import clear_output
from skimage import filters, measure
from skimage.color import label2rgb

import geometry, constants, commons


def build_graph(points, h=1, dp=1):
    """"""
    G = nx.Graph()
    for v in points.values():
        G.add_node(v)
    for k in points.keys():
        x, y = k
        if dp == 0:
            neighbors = [
                (int(x + 1), y),
                (int(x - 1), y),
                (x, int(y + 1)),
                (x, int(y - 1)),
            ]
        else:
            neighbors = [
                (round(x + h, dp), y),
                (round(x - h, dp), y),
                (x, round(y + h, dp)),
                (x, round(y - h, dp)),
            ]
        p0 = points[k]
        for neighbor in neighbors:
            p = points.get(neighbor)
            if p is None:
                continue
            G.add_edge(p0, p)

    return G


if __name__ == '__main__':
    img_files = sorted([os.path.join("oversegmented", f) for f in os.listdir("oversegmented") if f.endswith("tif")])
    for f in img_files:
        img_unseg = plt.imread(os.path.join("unsegmented", f.split("/")[-1]))
        img = cv2.imread(f)
        img_new = np.ones(img.shape[:2], dtype=int)
        for i in range(501):
            for j in range(501):
                v = img[i, j, :]
                if np.isclose(v, 255).all():
                    img_new[i, j] = 0
        
        points = {}
        counter = 0
        for idx in np.argwhere(np.isclose(img_new, 1)):
            points[tuple(idx)] = counter
            counter += 1 

        points_view = {v: k for k, v in points.items()}
        G = build_graph(points)
        pieces = nx.connected_components(G)
        pieces = [piece for piece in pieces]
        print("{:,} components".format(len(pieces)))
        img_seg = np.zeros(img_new.shape, dtype=np.uint8)
        valz = []
        # for idx, piece in enumerate(pieces):
        #     img_ = np.zeros(img.shape)
        #     coords = set()
        #     for p in piece:
        #         coord = tuple(points_view[p])
        #         coords.add(coord)
        #     avg = np.average(np.take(img_unseg, list(coords)))
        #     valz.append(avg)
        #     for coord in coords:
        #         img_[coord] = avg
        #     fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        #     ax[0].imshow(img_, cmap='gray')
        #     ax[0].set_title("Section")
        #     ax[1].imshow(img_unseg, cmap="gray")
        #     ax[1].set_title("Unsegmented")
        #     plt.show()
        #     phase = int(input("Enter phase: "))
        #     print(phase)
        #     np.put(img_seg, list(coords), phase)
        #     break
        fig, ax = plt.subplots(figsize=(5, 5))
        # qcs = ax.contour(img[:, :, 2], origin='image')
        # plt.clf()
        # print(qcs.levels)
        levels = [0, 40, 80, 120, 160, 200, 240, 280]
        ax.imshow(img_unseg,  cmap='gray')
        contours = measure.find_contours(img[:, :, 2], 0)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0])
        ax.set_title('Contour')
        ax.grid()
        plt.show()
        break