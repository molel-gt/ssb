#!/usr/bin/env python3
import cv2
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skimage as ski
from shapely.geometry import Polygon, Point
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon

import grapher, plot_opts, utils
plt.rcParams.update(plot_opts.params)


def get_curves(img_file_path, min_val):
    data = np.asarray(Image.open(img_file_path)).copy()
    img = data[:, :, 0]
    nx, ny = img.shape
    # pad boundaries
    img[:, 0] = 255
    img[:, -1] = 255
    img[0, :] = 255
    img[-1, :] = 255
    curves = np.where(np.greater_equal(img, min_val))

    return curves, nx, ny


def extract_voids_polygons(img):
    img = img - np.min(img)
    img_1 = ski.filters.gaussian(img, sigma=1.0) < 0.125
    contours = ski.measure.find_contours(img_1, 0.8)
    contours = sorted(contours, key=len, reverse=True)

    return contours


def points_in_polygon(polygon_coords, grid_size=1):
    """
    Generates discrete points within a polygon.

    Args:
        polygon_coords: A list of (x, y) tuples defining the polygon's vertices.
        grid_size: The spacing between grid points.

    Returns:
        A list of (x, y) tuples representing points inside the polygon.
    """
    polygon = Polygon(polygon_coords)
    minx, miny, maxx, maxy = polygon.bounds
    points = []

    for x in np.arange(minx, maxx + grid_size, grid_size):
        for y in np.arange(miny, maxy + grid_size, grid_size):
            point = Point(x, y)
            if polygon.contains(point):
                points.append((x, y))
    return points


def write_voids_polygon(voids_dir):
    for idx in range(1, 203):
        voids_path = os.path.join(voids_dir, f"{str(idx).zfill(3)}.json")
        print(f"Processing image {idx}")
        img = plt.imread(f"output/segmentation/raw/{str(idx).zfill(3)}.tif")
        voids_poly = [poly.tolist() for poly in extract_voids_polygons(img) if poly.shape[0] >= 4]
        with open(voids_path, "w", encoding='utf-8') as f:
            json.dump(voids_poly, f, ensure_ascii=False, indent=4)
    return


if __name__ == '__main__':
    voids_dir = os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/SEM Image/segmentation/voids")
    write_voids_polygon(voids_dir)
    # curves, nx, ny = get_curves("060.tif", 255)
    # img_1 = np.zeros((nx, ny))
    # img_1[curves] = 255
    # img_2 = Image.fromarray(img_1)
    # img_2.save("loops.tif")

    # # Find contours at a constant value of 0.8
    # contours = ski.measure.find_contours(img_1, 0.8)
    # print(f"There are {len(contours)} loops")

    # # Display the image and plot all contours found
    # fig, ax = plt.subplots()
    # ax.imshow(img_1, cmap=plt.cm.gray)
    # count = 0
    # # contour_sizes = [c.shape[0] for c in contours]
    # # perm = np.argsort(contour_sizes)
    # # perm.astype(int)
    # # print(perm)
    # contours = sorted(contours, key=len, reverse=True)
    # for contour in contours:
    #     print(contour.shape[0])
    #     # if contour.shape[0] <= 20:
    #     #       continue
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    #     count += 1
    # print(f"There are {count} big loops")

    # ax.axis('image')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.show()


    # points = {}
    # for (x, y) in curves:
    #     idx = ny * x + y
    #     points[idx] = (x, y) #=  idx
    # print(f"{len(points.keys())}")
    # fig, ax = plt.subplots()
    # ax.imshow(img, "gray")
    # # ax.scatter(curves[:, 0], curves[:, 1], s=0.1)
    # ax.set_box_aspect(1)
    # plt.tight_layout()
    # plt.show()
    # graph = grapher.PixelGraph(points)
    # graph.build_graph()
    # graph.get_graph_pieces()
    # print(f"Graph has {graph.n_pieces} pieces")
    # print(len(graph.pieces[-2]))
    # for p in graph.pieces[:50]:
    #     print(len(p))
