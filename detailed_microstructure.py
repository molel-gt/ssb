#!/usr/bin/env python3
import argparse
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


data_folder = os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/SEM Image/segmentation/")
fig, ax = plt.subplots()


class Segmentor:
    def __init__(self, img_id, root_folder=data_folder):
        self._img_id = img_id
        self._root_folder = data_folder
        self._img_raw = None
        self._cam_contours = []
        self._sse_contours = []

    @property
    def img_id(self):
        return self._img_id

    @property
    def root_folder(self):
        return self._root_folder

    @property
    def img_raw(self):
        return self._img_raw

    @property
    def cam_contours(self):
        return self._cam_contours

    @property
    def sse_contours(self):
        return self._sse_contours

    def setup(self):
        self._img_raw = plt.imread(os.path.join(self.root_folder, f"{str(self.img_id).zfill(3)}.tif"))

        return

    def get_contours(self):
        min_val = np.max(self.img_raw)
        img_file_path = os.path.join(self.root_folder, "edges", f"{str(self.img_id).zfill(3)}.tif")
        data = np.asarray(Image.open(img_file_path)).copy()
        img = data[:, :, 0]
        nx, ny = img.shape
        # pad boundaries
        img[:, 0] = 255
        img[:, -1] = 255
        img[0, :] = 255
        img[-1, :] = 255
        curves = np.where(np.greater_equal(img, min_val))
        # curves, nx, ny = get_curves("060.tif", 255)
        img_1 = np.zeros((nx, ny))
        img_1[curves] = 255
        contours = ski.measure.find_contours(img_1, 0.8)
        contours = sorted(contours, key=len, reverse=True)

        return contours

    def write_polygons_to_file(self):
        cam_polygons_path = os.path.join(self.root_folder, "cam", f"{str(self.img_id).zfill(3)}.json")
        sse_polygons_path = os.path.join(self.root_folder, "sse", f"{str(self.img_id).zfill(3)}.json")
        with open(cam_polygons_path, "w", encoding='utf-8') as f:
            json.dump(self.cam_contours, f, ensure_ascii=False, indent=4)

        with open(sse_polygons_path, "w", encoding='utf-8') as f:
            json.dump(self.sse_contours, f, ensure_ascii=False, indent=4)

        return


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


def write_voids_polygon(voids_dir, img_id):
    voids_path = os.path.join(voids_dir, f"{str(img_id).zfill(3)}.json")
    print(f"Processing image {idx}")
    img = plt.imread(f"output/segmentation/raw/{str(img_id).zfill(3)}.tif")
    voids_poly = [poly.tolist() for poly in extract_voids_polygons(img) if poly.shape[0] >= 4]
    with open(voids_path, "w", encoding='utf-8') as f:
        json.dump(voids_poly, f, ensure_ascii=False, indent=4)

    return


def on_press(event):
    print(event.key)
    if event.key == 'shift+right':
        ax.azim+=10
    if event.key == 'shift+left':
        ax.azim-=10
    fig.canvas.draw_idle()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='secondary current distribution')
    parser.add_argument('--img_id', help='`image index` to process', required=True)
    parser.add_argument("--extract_voids", help="whether to extract voids", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    voids_dir = os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/SEM Image/segmentation/voids")
    cam_dir = os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/SEM Image/segmentation/cam")
    sse_dir = os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/SEM Image/segmentation/sse")
    raw_dir = os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/SEM Image/segmentation/raw")
    if args.extract_voids:
        write_voids_polygon(voids_dir, args.img_id)
    curves, nx, ny = get_curves("060.tif", 255)
    img_1 = np.zeros((nx, ny))
    img_1[curves] = 255
    # img_2 = Image.fromarray(img_1)
    # img_2.save("loops.tif")

    # Find contours at a constant value of 0.8
    contours = ski.measure.find_contours(img_1, 0.8)
    contours = sorted(contours, key=len, reverse=True)
    print(f"There are {len(contours)} loops")
    img_id = 60
    img_raw = plt.imread(os.path.join(raw_dir, f"{str(img_id).zfill(3)}.tif"))
    sse_polys = []
    cam_polys = []

    # # Display the image and plot all contours found
    ax.imshow(img_raw, "gray")
    for contour in contours:
        if contour.shape[0] < 4:
            continue
        ax.scatter(contour[:, 1], contour[:, 0], s=0.5, color="red")
        fig.canvas.draw_idle()

    plt.show()

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
