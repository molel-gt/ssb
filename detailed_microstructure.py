#!/usr/bin/env python3
import argparse
import json
import os

import bezier
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import shapely

import skimage as ski
# from matplotlib.patches import Polygon
from PIL import Image
from shapely import plotting
from shapely.geometry import Polygon, Point
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon

import geometry, grapher, plot_opts, utils
# plt.rcParams.update(plot_opts.params)


data_folder = os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/SEM Image/segmentation/")
# fig, ax = plt.subplots()


class Segmentor:
    def __init__(self, img_id, ax, fig, root_folder=data_folder):
        self._img_id = img_id
        self._root_folder = data_folder
        self._img_raw = None
        self._cam_contours = []
        self._sse_contours = []
        self._tbd_contours = []
        self._current_phase = 0
        self._fig = fig
        self._ax = ax

    @property
    def img_id(self):
        return self._img_id

    @property
    def ax(self):
        return self._ax

    @property
    def fig(self):
        return self._fig

    @property
    def root_folder(self):
        return self._root_folder

    @property
    def current_phase(self):
        return self._current_phase

    @property
    def img_raw(self):
        return self._img_raw

    @property
    def cam_contours(self):
        return self._cam_contours

    @property
    def sse_contours(self):
        return self._sse_contours

    @property
    def tbd_contours(self):
        return self._tbd_contours

    def setup(self):
        self._img_raw = plt.imread(os.path.join(self.root_folder, "raw", f"{str(self.img_id).zfill(3)}.tif"))

        return

    def get_contours(self):
        min_val = 125#np.max(self.img_raw)
        img_file_path = os.path.join(self.root_folder, "edges", f"{str(self.img_id).zfill(3)}.tif")

        loops_img_file_path = os.path.join(self.root_folder, "loops", f"{str(self.img_id).zfill(3)}.tif")
        data = np.asarray(Image.open(img_file_path)).copy()
        img = data[:, :, 0]
        nx, ny = img.shape
        # pad boundaries
        # img[:, 0] = 255
        # img[:, -1] = 255
        # img[0, :] = 255
        # img[-1, :] = 255
        curves = np.where(np.greater_equal(img, min_val))
        img_1 = np.zeros((nx, ny), dtype=np.uint8)
        img_1[curves] = 255
        # img_1 = np.greater_equal(data[:, :, 0], data[:, :, 3])
        contours = ski.measure.find_contours(img_1, 0)#, fully_connected="high")#, "low", "low")
        contours = sorted(contours, key=len, reverse=True)

        return contours

    def write_polygons_to_file(self):
        cam_polygons_path = os.path.join(self.root_folder, "cam", f"{str(self.img_id).zfill(3)}.json")
        sse_polygons_path = os.path.join(self.root_folder, "sse", f"{str(self.img_id).zfill(3)}.json")
        tbd_polygons_path = os.path.join(self.root_folder, "tbd", f"{str(self.img_id).zfill(3)}.json")
        with open(cam_polygons_path, "w", encoding='utf-8') as f:
            json.dump(self.cam_contours, f, ensure_ascii=False, indent=4)

        with open(sse_polygons_path, "w", encoding='utf-8') as f:
            json.dump(self.sse_contours, f, ensure_ascii=False, indent=4)

        with open(tbd_polygons_path, "w", encoding='utf-8') as f:
            json.dump(self.tbd_contours, f, ensure_ascii=False, indent=4)

        return

    def on_press(self, event):
        print(event.key)
        if event.key == '2':
            self._current_phase = 2
        elif event.key == '1':
            self._current_phase = 1
        else:
            self._current_phase = 0
        self.fig.canvas.draw_idle()


def extract_voids_polygons(img):
    img = img - np.min(img)
    img_1 = ski.filters.gaussian(img, sigma=1.0) < 0.125
    contours = ski.measure.find_contours(img_1, 0)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='segmentation')
    parser.add_argument('--img_id', help='id number of image to process', required=True)
    parser.add_argument("--extract_voids", help="whether to extract voids", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    voids_dir = os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/SEM Image/segmentation/voids")
    cam_dir = os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/SEM Image/segmentation/cam")
    sse_dir = os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/SEM Image/segmentation/sse")
    raw_dir = os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/SEM Image/segmentation/raw")
    if args.extract_voids:
        write_voids_polygon(voids_dir, args.img_id)
    fig, ax = plt.subplots()
    segmentor = Segmentor(img_id=args.img_id, ax=ax, fig=fig)
    segmentor.setup()
    ax.imshow(segmentor.img_raw, "gray")
    fig.canvas.mpl_connect('key_press_event', segmentor.on_press)
    ax.set_xlim([0, segmentor.img_raw.shape[0]])
    ax.set_ylim([0, segmentor.img_raw.shape[1]])
    ax.set_box_aspect(1)
    fig.canvas.manager.resize(*fig.canvas.manager.window.maxsize())
    poly_plot = None
    cam_coords = []
    sse_coords = []
    pending_coords = []
    wrote_data = False
    for contour in segmentor.get_contours()[1:]:
        coords = ski.measure.approximate_polygon(contour, tolerance=1)
        # coords = np.vstack((coords, coords[0, :]))
        if len(coords) < 4:
            continue
        polygon = Polygon([(c[1], c[0]) for c in coords])
        if np.isclose(polygon.area, 0):
            continue
        if polygon.area < 100:
            continue
        hull_cc = shapely.concave_hull(polygon, ratio=0.1)
        hull_cv = shapely.convex_hull(polygon)
        if poly_plot is not None:
            poly_plot.remove()
        coords_arr = np.array(coords)
        # poly_plot = ax.scatter(coords[:, 1], coords[:, 0], s=0.5, linestyle="-.", color="white")
        poly_plot = plotting.plot_polygon(polygon, ax=ax, linewidth=0.5, add_points=False, facecolor=None, edgecolor="red")
        keypressed = False
        while not keypressed:
            keypressed = plt.waitforbuttonpress()
        if segmentor.current_phase == 1:
            segmentor.sse_contours.append(coords.tolist())
        elif segmentor.current_phase == 2:
            segmentor.cam_contours.append(coords.tolist())
        else:
            segmentor.tbd_contours.append(coords.tolist())
    segmentor.write_polygons_to_file()
    wrote_data = True
    segmentor.cam_contours.clear()
    segmentor.sse_contours.clear()
    segmentor.tbd_contours.clear()
        # nodes = np.asfortranarray([coords[:, 1].tolist(), coords[:, 0].tolist()])
        # curve = bezier.Curve(nodes)
    plt.tight_layout()
    plt.show()
    if wrote_data:
        quit()
