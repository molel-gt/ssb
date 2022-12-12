#! /usr/bin/env python3

import csv
import os
import subprocess
import time

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from IPython.display import clear_output
from skimage import filters, measure
from skimage.color import label2rgb

import geometry, constants, commons

# atomics masses of elements used in the composite cathode
atomic_masses = {
    "Li": 6.94,
    "O": 15.999,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Mn": 54.938,
    "Co": 58.933,
    "Ni": 58.693,
}

atomic_ratios_nmc622 = {
    "Li": 0.25,
    "O": 0.5,
    "Mn": 0.05,
    "Ni": 0.15,
    "Co": 0.05,
    "S": 0,
    "P": 0,
    "Cl": 0,

}

atomic_ratios_lpsc = {
    "Li": 6/13,
    "O": 0,
    "Mn": 0,
    "Ni": 0,
    "Co": 0,
    "S": 5/13,
    "P": 1/13,
    "Cl": 1/13,
}

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
        print(np.unique(img))

        fig, ax = plt.subplots(figsize=(5, 5))
        # qcs = ax.contour(img[:, :, 2], origin='image')
        # plt.clf()
        # print(qcs.levels)
        levels = [0, 40, 80, 120, 160, 200, 240, 280]
        ax.imshow(img_unseg,  cmap='gray')
        img_clusters = np.ones(img.shape[:2])
        contours = measure.find_contours(img[:, :, 2], 0)
        for contour in contours:
            for coord in contour:
                img_clusters[tuple([int(v) for v in coord])] = 0
            ax.plot(contour[:, 1], contour[:, 0])
        # ax.imshow(img_clusters, cmap='gray')
        # ax.set_aspect('equal', 'box')
        # ax.set_title('Contour')
        # ax.grid()
        # plt.savefig("figures/superpixels.png")
        plt.savefig("figures/superpixels.png", format='png', edgecolor='red', bbox_inches='tight', pad_inches=0)
        plt.show()
    
        points = {}
        counter = 0
        for idx in np.argwhere(np.isclose(img_clusters, 1)):
            points[tuple(idx)] = counter
            counter += 1 

        points_view = {v: k for k, v in points.items()}
        G = build_graph(points)
        pieces = nx.connected_components(G)
        pieces = [piece for piece in pieces]
        print("{:,} components".format(len(pieces)))
        img_seg = np.copy(img_unseg)  # np.zeros(img_unseg.shape, dtype=np.uint8)
        valz = []
        with open("data/clusters.csv", "w") as fp:
            writer = csv.DictWriter(fp, fieldnames=["centroid", "pixel_avg", "pixel_std"])
            writer.writeheader()
            for idx, piece in enumerate(pieces):
                coords = set()
                for p in piece:
                    coord = tuple(points_view[p])
                    coords.add(coord)
                avg = np.average([img_unseg[c] for c in coords])
                xc = np.average([c[0] for c in coords])
                yc = np.average([c[1] for c in coords])
                std = np.std([img_unseg[c] for c in coords])
                writer.writerow({"centroid": f"({xc:.2f}, {yc:.2f})", "pixel_avg": f"{avg:.2f}", "pixel_std": f"{std:.2f}"})
                print(f"{avg:.2f}, {std:.2f}")
                
                valz.append(avg)
                for coord in coords:
                    # img_seg[coord] = std
                    img_seg[coord] = avg
        # fig, ax = plt.subplots(figsize=(10, 10))
        # ax.imshow(img_seg, cmap='gray')
        # ax.set_title("Section")
        # ax.set_axis_off()
        # ax.imshow(img_unseg, cmap="gray")
        # ax.set_title("Unsegmented")
        # plt.savefig("figures/img-clusters.tif", format='TIF', bbox_inches='tight', pad_inches=0)
        # plt.show()
        # phase = int(input("Enter phase: "))
        # print(phase)
        # np.put(img_seg, list(coords), phase)
        break
        