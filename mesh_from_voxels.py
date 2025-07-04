#!/usr/bin/env python3
import json
import os

import gmsh
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from PIL import Image
from shapely.geometry import Polygon, Point

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


if __name__ == '__main__':
    # img = plt.imread("/home/lmolel/OneDrive/PhD/Data/FIB-SEM JG/Archive/4.tif46.tif116.tif105.tif102.tif")
    # print(img.shape)
    # img_m = ski.filters.meijering(img[:, :, 0]) * 255
    # contours = ski.measure.find_contours(img_m, 0)
    # fig, ax = plt.subplots()
    # ax.imshow(img[:, :, 0], "gray")
    # print(len(contours))
    # for contour in contours:

    #     coords = ski.measure.approximate_polygon(contour, tolerance=0.1)
    #     if len(coords) < 4:
    #         continue
    #     polygon = Polygon([(c[1], c[0]) for c in coords])
    #     if polygon.area < 400:
    #         continue
    #     ax.plot(coords[:, 1], coords[:, 0], "r--")
    # plt.show()
    cam_dir = os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/SEM Image/segmentation/cam")
    voids_dir = os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/SEM Image/segmentation/voids")
    for img_id in range(1, 203):
        out_imgfile = os.path.join(f"output/segmentation/cam/{str(img_id).zfill(3)}.tif")
        surfs = []
        points = []
        out_img = np.zeros((500, 500), dtype=np.uint8)
        with open(os.path.join(cam_dir, f"{str(img_id).zfill(3)}.json"), "r") as fp:
            data = json.load(fp)
        counter = 0
        for row in data:
            counter += 1
            poly_points = []
            poly_lines = []
            seen_points = set()
            p_count = 0
            points = [(x, y) for (x, y) in row]
            poly_points = points_in_polygon(points)
            for (x, y) in poly_points:
                out_img[int(x), int(y)] = 2
        img = Image.fromarray(out_img)
        img.save(out_imgfile)
        # fig, ax = plt.subplots()
        # ax.imshow(out_img, "gray")
        # ax.set_box_aspect(1)
        # plt.tight_layout()
        # plt.show()

