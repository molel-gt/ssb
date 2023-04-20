#!/usr/bin/env python3
import datetime
import itertools
import os
import time

import alphashape
import argparse
import cartopy.crs as ccrs
import cv2
import hdbscan
import igraph as ig
import matplotlib.pyplot as plt
import metis
import networkx as nx
import numpy as np


import pickle
import warnings

from concavehull import concavehull
from igraph import Graph
from shapely import Polygon, MultiPoint
from shapely.plotting import plot_polygon
from shapely.validation import make_valid
from descartes import PolygonPatch
from ipywidgets import widgets, interactive
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patches
from matplotlib.widgets import CheckButtons, Button, Slider, LassoSelector, RadioButtons, TextBox, RectangleSelector
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from skimage import filters
from sklearn.ensemble import RandomForestClassifier

from hulls import ConcaveHull

import geometry


warnings.simplefilter("ignore")


rerun = False
image = None
NX = 501
NY = 501
NZ = 202
hdbscan_kwargs = {
    "min_cluster_size": 25,
    "cluster_selection_epsilon": 5,
    "gen_min_span_tree": True,
    'cluster_selection_method': 'leaf',
    'min_samples': 5,
    }

phases = {
    "None": -1,
    "Void": 0,
    "Solid Electrolyte": 1,
    "Active Material": 2,
    }

training_images = np.linspace(0, 200, num=41)
thresholds = ['-0.80', '-0.50','-0.40', '-0.30', '-0.20', '-0.01', '0.00', '0.01', '0.02','0.03', '0.04', '0.05', '0.10', '0.20', '0.30', '0.40', '0.50', '0.99']


class PixelGraph:
    def __init__(self, points):
        self._points = points
        self._pieces = None
        self._graph = None
        self._n_pieces = 0

    @property
    def points(self):
        return self._points

    @property
    def pieces(self):
        return self._pieces

    @property
    def n_pieces(self):
        return self._n_pieces

    @property
    def graph(self):
        return self._graph

    def build_graph(self):
        n_nodes = max(self.points.keys())

        points_lookup = {v: k for k, v in self.points.items()}
        edges = []
        for idx, p in self.points.items():
            x, y = p
            neighbors = [(int(x), int(y+1)), (int(x+1), int(y)), (int(x+1), int(y+1))]
            for n in neighbors:
                n_idx = points_lookup.get(n)
                if n_idx is None:
                    continue
                edges.append((idx, n_idx))

        G = ig.Graph(n_nodes + 1, edges)

        self._graph = G

    def get_graph_pieces(self):
        self._pieces = list(self.graph.connected_components())
        self._n_pieces = len(self.pieces)


def make_dir_if_missing(f_path):
    """"""
    os.makedirs(f_path, exist_ok=True)


def load_clusters(X_2d, y_predict, shape=(501, 501)):
    img_seg = -2 * np.ones(shape)
    for v in np.unique(y_predict):
        X_v = np.argwhere(y_predict == v)
        coords = list(itertools.chain.from_iterable(X_v))
        for coord in coords:
            xx, yy = X_2d[int(coord), :2]
            img_seg[int(xx), int(yy)] = v
    return img_seg


def enhance_clusters(img_seg):
    new_img = -2 * np.ones(img_seg.shape)
    for i in range(img_seg.shape[0]):
        for j in range(img_seg.shape[1]):
            v = img_seg[i, j]
            if v != -2:
                new_img[i, j] = v
                continue
            neighs = [
                (i, j + 1),
                (i + 1, j),
                (i, j - 1),
                (i - 1, j),
                (i + 1, j + 1),
                (i - 1, j - 1),
                (i - 1, j + 1),
                (i + 1, j - 1),
                ]
            vals = []
            for neigh in neighs:
                try:
                    vals.append(img_seg[neigh])
                except IndexError:
                    continue
            set_vals = set(vals)
            if -2 in set_vals:
                set_vals.remove(-2)
            if len(set_vals) == 1:
                new_img[i, j] = set_vals.pop()
    
    return new_img


def neighborhood_average(arr, d=(1, 1), n_min=(0, 0), n_max=(501, 501)):
    out = np.zeros(arr.shape)
    dx, dy = d
    for i in range(n_max[0]):
        for j in range(n_max[1]):
            neighbors = arr[max(i - dx, 0):min(i + dx, n_max[0] - 1), max(j - dy, 0):min(j + dy, n_max[1] - 1)]
            out[i, j] = np.mean(neighbors)
    return out


def build_features_matrix(img, img_1, threshold):
    """"""
    if threshold < 0:
        coords = np.asarray(np.where(np.logical_and(np.greater_equal(1 - img, 0), np.less_equal(1 - img, np.abs(threshold))))).T
    else:
        coords = np.asarray(np.where(np.logical_and(np.greater_equal(img, 0), np.less_equal(img, threshold)))).T
    y = np.array([img_1[ix, iy] for (ix, iy) in coords]).reshape(-1, 1) / 255
    y_ = np.array([img[ix, iy] for (ix, iy) in coords]).reshape(-1, 1)
    X_2d = np.hstack((coords, y, y_))

    return X_2d


def chunk_array(arr_shape, arr, size=100):
    x_points = list(range(0, arr_shape[0], size))
    y_points = list(range(0, arr_shape[1], size))
    if arr_shape[0] > x_points[-1] + 1:
        x_points += [-1]
    if arr_shape[1] > y_points[-1] + 1:
        y_points += [-1]
    for ix, x in enumerate(x_points[:-1]):
        for iy, y in enumerate(y_points[:-1]):
            coords = np.where(
                np.logical_and(
                    np.logical_and(np.greater_equal(arr[:, 0], x), np.less(arr[:, 0], x_points[ix+1])),
                    np.logical_and(np.greater_equal(arr[:, 1], y), np.less(arr[:, 1], y_points[iy+1])),
                    )
                )
            yield coords


def get_clustering_results(X_2d, **hdbscan_kwargs):
    clusterer = hdbscan.HDBSCAN(**hdbscan_kwargs)
    y_predict = clusterer.fit_predict(X_2d).reshape(-1, 1)
    # y_predict = np.zeros((X_2d.shape[0], ), dtype=np.intc)
    # max_cluster_id = 0
    # for coords in chunk_array((501, 501), X_2d):
    #     features = X_2d[coords[0], :]
    #     predictions = clusterer.fit_predict(features)
    #     new_c = np.where(predictions < 0)
    #     new_c2 = np.where(predictions > -1)
    #     predictions[new_c2] = predictions[new_c2] + max_cluster_id
    #     y_predict[coords] = predictions
    #     max_cluster_id = np.max(y_predict) + 1

    return y_predict

new_plot = None
def get_polygon(clusters, ax):
    max_v = np.max(clusters)
    new_clusters = clusters.copy()
    adder = 0
    for v in np.unique(clusters):
        if v < 0:
            continue

        coords = np.where(np.isclose(clusters, v))
        points = [(coords[1][i], coords[0][i]) for i in range(coords[0].shape[0])]
        points_arr = np.array(points).reshape(-1, 2)
        points_dict = {}
        for i in range(coords[0].shape[0]):
            points_dict[int(i)] = (int(coords[1][i]), int(coords[0][i]))

        PG = PixelGraph(points=points_dict)
        PG.build_graph()
        PG.get_graph_pieces()
        pieces = PG.pieces
        n_pieces = PG.n_pieces
        if np.isclose(n_pieces, 1):
            hull = concavehull(points_arr, chi_factor=1e-12)
            ax.plot(hull[:, 0] - 5, hull[:, 1] - 5, 'w--', linewidth=0.5)
        else:
            for i, p in enumerate(pieces):
                p_points = [points_dict[idx] for idx in p]
                p_points_arr = np.array(p_points).reshape(-1, 2)
                if i > 0:
                    adder += 1
                    new_clusters[p_points] = max_v + adder
                try:
                    hull = concavehull(p_points_arr, chi_factor=1e-12)
                    # polygon = Polygon(p_points)
                    ax.plot(hull[:, 0] - 5, hull[:, 1] - 5, 'w--', linewidth=0.5)
                except RuntimeError:
                    print("Cannot triangulate", v, i)


class Segmentor:
    def __init__(self, image, image_id=0, threshold=0.10, output_dir='segmentation'):
        self.image_id = image_id
        self.threshold = threshold
        self._output_dir = output_dir
        self.image = image
        self._clusters = -2 * np.ones(image.shape) 
        self.edges = None
        self._phases = np.ones(self.image.shape, dtype=np.intc)
        self.residual = -1 * np.ones(self.image.shape, dtype=np.intc)
        self.rerun = False
        self.use_residuals = False

    @property
    def output_dir(self):
        return self._output_dir

    @property
    def clusters(self):
        return self._clusters

    @property
    def phases(self):
        return self._phases

    @property
    def edges_dir(self):
        return os.path.join(self.output_dir, 'edges')

    @property
    def clusters_dir(self):
        return os.path.join(self.output_dir, 'clusters')

    @property
    def phases_dir(self):
        return os.path.join(self.output_dir, 'phases')

    def create_dirs(self):
        make_dir_if_missing(self.output_dir)
        make_dir_if_missing(self.edges_dir)
        make_dir_if_missing(self.clusters_dir)
        make_dir_if_missing(self.phases_dir)

    def update_residuals(self):
        coords = np.where(self.phases != 1)
        self.residual[coords] = self.image[coords]

    def update_phases(self, selection, phase):
        self._phases[selection] = phase
        self._clusters[selection] = -2
        self.update_residuals()

        with open(os.path.join(self.phases_dir, f'{str(self.image_id).zfill(3)}'), 'wb') as fp:
            pickle.dump(self.phases, fp)

    def set_edges(self):
        if not self.rerun and os.path.exists(os.path.join(self.edges_dir, f'{str(self.image_id).zfill(3)}')):
            with open(os.path.join(self.edges_dir, f'{str(self.image_id).zfill(3)}'), 'rb') as fp:
                self.edges = pickle.load(fp)
        else:
            img_11 = neighborhood_average(self.image)
            for i in range(5):
                img_11 = neighborhood_average(img_11)
            img_2 = filters.gaussian(img_11, sigma=0.5)
            img_2 = neighborhood_average(img_2 / np.max(img_2))
            img_2 = filters.meijering(img_2 / np.max(img_2))
            # img_2 = neighborhood_average(img_2)
            self.edges = img_2 / np.max(img_2)
            self.write_edges_to_file()
            # with open(os.path.join(self.edges_dir, f'{str(self.image_id).zfill(3)}'), 'wb') as fp:
            #     pickle.dump(self.edges, fp)

    def write_edges_to_file(self):
        with open(os.path.join(self.edges_dir, f'{str(self.image_id).zfill(3)}'), 'wb') as fp:
            pickle.dump(self.edges, fp)

    def clustering(self):
        if os.path.exists(os.path.join(self.phases_dir, f'{str(self.image_id).zfill(3)}')):
            with open(os.path.join(self.phases_dir, f'{str(self.image_id).zfill(3)}'), 'rb') as fp:
                self._phases = pickle.load(fp)
        else:
            self._phases = np.ones(self.image.shape, dtype=np.intc)

        self.set_edges()

        img = self.edges
        if self.use_residuals:
            coords = np.where(self.phases != 1)
            img[coords] = -1
        X_2d = build_features_matrix(img, self.image, self.threshold)
        if not np.all(np.array(X_2d.shape) > 0):
            return
        y_predict = get_clustering_results(X_2d, **hdbscan_kwargs)
        img_cluster_raw = -2 * np.ones(img.shape)  # -2, -1 are residual non-clustered

        for v in np.unique(y_predict):
            if v < 0:
                continue
            X_v = np.where(y_predict == v)[0]
            coords = np.array([X_2d[ix, :2] for ix in X_v])
            for (ix, iy) in coords:
                img_cluster_raw[int(ix), int(iy)] = int(v)

        img_cluster_enhanced = enhance_clusters(img_cluster_raw)

        self._clusters = img_cluster_enhanced

    def run(self, selection=None, phase=None, rerun=False, clustering=False, segmentation=False, use_residuals=True):
        self.rerun = rerun
        self.use_residuals = use_residuals
        self.create_dirs()

        if clustering:
            self.clustering()

        if segmentation:
            self.update_phases(selection, phase)


class App:
    def __init__(self, seg, selected_phase=-1, f_ind=0, t_ind=10, fs=None, fig=None, radio=None, ax=None):
        self.seg = seg
        self.ind = f_ind
        self._selected_phase = selected_phase
        self._threshold_index = t_ind
        self._fs = fs
        self._fig = fig
        self._ax = ax

    @property
    def image_id(self):
        return int(training_images[self.ind])

    @property
    def threshold(self):
        return float(thresholds[int(self._threshold_index)])

    @property
    def selected_phase(self):
        return self._selected_phase

    def next(self, event):
        self.ind += 1
        self.ind = int(self.ind)
        fig.suptitle(f"File: unsegmented/{str(self.image_id).zfill(3)}.tif")
        with open(os.path.join('unsegmented', str(self.image_id).zfill(3) + '.tif'), 'rb') as fp:
            image = plt.imread(fp)

        self.seg.image = image
        self.seg.image_id = int(self.image_id)
        self.seg.threshold = self.threshold
        self.seg.run(rerun=False, clustering=True)
        f1, f2, f3, f4 = self._fs

        f1.set_data(image)
        f2.set_data(self.seg.edges)
        f3.set_data(self.seg.clusters)
        f4.set_data(self.seg.phases)
        get_polygon(self.seg.clusters, self._ax[0, 0])
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def prev(self, event):
        self.ind -= 1
        self.ind = int(self.ind)
        fig.suptitle(f"File: unsegmented/{str(self.image_id).zfill(3)}.tif")
        with open(os.path.join('unsegmented', str(self.image_id).zfill(3) + '.tif'), 'rb') as fp:
            image = plt.imread(fp)

        self.seg.image = image
        self.seg.image_id = int(self.image_id)
        self.seg.threshold = self.threshold
        self.seg.run(rerun=False, clustering=True)
        f1, f2, f3, f4 = self._fs
        f1.set_data(image)
        f2.set_data(self.seg.edges)
        f3.set_data(self.seg.clusters)
        f4.set_data(self.seg.phases)
        get_polygon(self.seg.clusters, self._ax[0, 0])
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def onSelect(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        selected_pts = []
        for ix in range(x1, x2 + 1):
            for iy in range(y1, y2 + 1):
                selected_pts.append((ix, iy))
        selected_pts = np.array(selected_pts, dtype=int)
        cluster_vals = [int(v) for v in np.unique([self.seg.clusters[iy, ix] for ix, iy in selected_pts]) if v > -1]

        f1, f2, f3, f4 = self._fs
        for v in cluster_vals:
            coords = np.where(self.seg.clusters == v)
            self.seg.run(selection=coords, phase=self.selected_phase, segmentation=True)

            f3.set_data(self.seg.clusters)      
            f4.set_data(self.seg.phases)
            get_polygon(self.seg.clusters, self._ax[0, 0])
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    def newEdges(self, val):
        selection = list(set([(int(x), int(y)) for x, y in val]))
        select = (np.array([x[0] for x in selection]).reshape(-1, 1), np.array([y[1] for y in selection]).reshape(-1, 1))
        self.seg.edges[select] = 1
        self.seg.write_edges_to_file()
        self.seg.run(rerun=False, clustering=True)
        f1, f2, f3, f4 = self._fs
        f2.set_data(self.seg.edges)
        f3.set_data(self.seg.clusters)
        get_polygon(self.seg.clusters, self._ax[0, 0])
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def onCorrect(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        selected_pts = []
        count = 0
        for ix in range(x1, x2 + 1):
            for iy in range(y1, y2 + 1):
                selected_pts.append((ix, iy))
                count += 1
        selected_pts = np.array(selected_pts, dtype=int).reshape(count, 2)
        selection = (selected_pts[:, 1], selected_pts[:, 0])
        self.seg.run(selection=selection, phase=self.selected_phase, segmentation=True)

        f1, f2, f3, f4 = self._fs
        f3.set_data(self.seg.clusters)      
        f4.set_data(self.seg.phases)
        get_polygon(self.seg.clusters, self._ax[0, 0])
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def check_threshold(self, label):
        # uncheck previous
        index = thresholds.index(label)
        if index != int(self._threshold_index):
            check.set_active(int(self._threshold_index))
        self._threshold_index = index

        self.seg.threshold = self.threshold
        self.seg.run(rerun=False, clustering=True)

        f2.set_data(self.seg.edges)
        f3.set_data(self.seg.clusters)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    def switch_threshold(self, val):
        self.threshold = val
        self.seg.threshold = self.threshold
        self.seg.run(rerun=False, clustering=True)
        f1, f2, f3, f4 = self._fs

        f2.set_data(self.seg.edges)
        f3.set_data(self.seg.clusters)
        get_polygon(self.seg.clusters, self._ax[0, 0])
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def select_phase(self, val):
        self._selected_phase = phases[val]


class StackSeg:
    def __init__(self, training_images, testing_images=None, output_dir='2023-03-04'):
        self._model = RandomForestClassifier(n_jobs=8,
                                                criterion='gini',
                                                oob_score=True,
                                                class_weight='balanced_subsample',
                                                # n_estimators=500,
                                                # bootstrap=False,
                                                )
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None
        self._y_test_pred = None
        self._X_validate = None
        self._y_validate = None
        self._training_images = training_images
        self._data = None
        self._output_dir = output_dir

    @property
    def model(self):
        return self._model

    @property
    def X_train(self):
        return self._X_train

    @property
    def X_test(self):
        return self._X_test

    @property
    def X_validate(self):
        return self._X_validate

    @property
    def y_train(self):
        return self._y_train

    @property
    def y_test(self):
        return self._y_test

    @property
    def y_test_pred(self):
        return self._y_test_pred

    @property
    def y_validate(self):
        return self._y_validate

    @property
    def data(self):
        return self._data

    @property
    def training_images(self):
        return self._training_images

    @property
    def output_dir(self):
        return self._output_dir

    def build_features_matrix(self):
        self._data = np.zeros((501 * 501 * self.training_images.size, 5), dtype=np.intc)
        train_data = np.zeros((0, 5), dtype=np.intc)
        test_data = np.zeros((0, 5), dtype=np.intc)

        for img_no in self.training_images:
            print(f"Loading image {int(img_no)}")
            raw_img = plt.imread(f'unsegmented/{str(int(img_no)).zfill(3)}.tif')
            # raw_img = (raw_img / 255) * filters.meijering(raw_img)
            # raw_img = filters.gaussian(raw_img)
            with open(f'{self.output_dir}/phases/{str(int(img_no)).zfill(3)}', 'rb') as fp:
                image = pickle.load(fp)
                coords = np.where(image > -1)
                rows = np.zeros((coords[0].size, 5), dtype=np.intc)
                rows[:, 0] = coords[0]
                rows[:, 1] = coords[1]
                rows[:, 2] = img_no * np.ones(coords[0].shape)
                rows[:, 3] = raw_img[coords] / 255
                rows[:, 4] = image[coords]
                if int(int(img_no) % 10) == 0:
                    train_data = np.vstack((train_data, rows))
                else:
                    test_data = np.vstack((test_data, rows))
        print("Building features..")
        self._X_train = train_data[:, :4]
        self._y_train = train_data[:, 4].reshape(-1, 1)
        self._X_test = test_data[:, :4]
        self._y_test = test_data[:, 4].reshape(-1, 1)
        print("Built features.")

    def train(self):
        print("Training..")
        self.model.fit(self.X_train, self.y_train)
        print("Training Score:", self.model.score(self.X_train, self.y_train))

    def validate(self):
        self._y_validate = self.model.predict(self.X_validate)

    def test(self):
        print("Testing..")
        self._y_test_pred = self.model.predict(self.X_test)
        print("Testing Score:", self.model.score(self.X_test, self.y_test))

    def retrain(self):
        X_train = np.vstack((self.X_train, self.X_test))
        y_train = np.vstack((self.y_train, self.y_test))
        print("Starting model training..")
        self.model.fit(X_train, y_train)
        print("Done training model")

    def create_output(self):
        print("Retraining Model")
        start = time.time()
        self.build_features_matrix()
        self.retrain()
        stop = time.time()
        print("Training took {:,} minutes".format(int(int(stop - start)/60)))
        for z in range(NZ):
            print(f"Segmenting image {z}")
            img =  plt.imread(f'unsegmented/{str(int(z)).zfill(3)}.tif')
            img_11 = neighborhood_average(img)
            for i in range(5):
                img_11 = neighborhood_average(img_11)
            img = img_11 / 255
            features = np.zeros((NX * NY, 4))
            coords = np.where(img > -1)
            features[:, 0] = coords[0]
            features[:, 1] = coords[1]
            features[:, 2] = int(z)
            features[:, 3] = img[coords]
            output = self.model.predict(features)
            img_out = np.zeros(img.shape, dtype=np.uint8)
            img_out[coords] = output
            # new_img = Image.fromarray(img_out, mode='P')
            cv2.imwrite(f'segmented/{str(int(z)).zfill(3)}.tif', img_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation')
    parser.add_argument('--output_dir', help='working directory for output, e.g. YYYY-mm-dd', default=datetime.datetime.now().strftime('%Y-%m-%d'))

    args = parser.parse_args()
    fig, ax = plt.subplots(2, 3)
    fig.subplots_adjust(left=0)
    ax[0, 0].grid(which='both')
    ax[1, 0].grid(which='both')
    ax[0, 1].grid(which='both')
    ax[1, 1].grid(which='both')
    ax[0, 2].set_axis_off()
    ax[1, 2].set_axis_off()

    image_id = 0
    with open(os.path.join('unsegmented', str(image_id).zfill(3) + '.tif'), 'rb') as fp:
        image = plt.imread(fp)

    seg = Segmentor(image, image_id=image_id, threshold=float(thresholds[12]), output_dir=args.output_dir)
    seg.run(rerun=False, clustering=True)#, use_residuals=False)
    fig.suptitle(f"Image: unsegmented/{str(image_id).zfill(3)}.tif")

    axcolor = 'lightgoldenrodyellow'
    rax = inset_axes(ax[0, 2], width="100%", height='70%', loc=3)
    rax.set_facecolor(axcolor)

    # checkbuttons
    check = CheckButtons(ax[1, 2], thresholds, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    # next and previous buttons
    axprev = inset_axes(ax[0, 2], width="49.5%", height='10%', loc=2)
    axnext = inset_axes(ax[0, 2], width="49.5%", height='10%', loc=1)
    cmap_name = 'seg'
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    cdict = {
        'gray': (
            (0.0,  0.0, 0.0),
            (1.0,  1.0, 1.0),
            (1.0,  1.0, 1.0),
            ),
        'red': (
            (0.0,  0.0, 0.0),
            (0.5,  1.0, 1.0),
            (1.0,  1.0, 1.0),
            ),
        'green': (
            (0.0,  0.0, 0.0),
            (0.25, 0.0, 0.0),
            (0.75, 1.0, 1.0),
            (1.0,  1.0, 1.0),
        ),
        'blue': (
            (0.0,  0.0, 0.0),
            (0.5,  0.0, 0.0),
            (1.0,  1.0, 1.0),
        )
        }

    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=4)

    radio = RadioButtons(
        rax,
        ('None', 'Void', 'Solid Electrolyte', 'Active Material'),
        active=0,
        label_props={'color': ['gray', 'blue', 'red' , 'green']},
        radio_props={'edgecolor': ['gray', 'darkblue', 'darkred', 'darkgreen'],
                      'facecolor': ['gray', 'blue', 'red', 'green'],
                      },
        )

    f1 = ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].set_title('Original')
    ax[0, 0].set_aspect('equal', 'box')
    # ax[0, 0].set_xlim([0, 501])
    # ax[0, 0].set_ylim([0, 501])

    fig.canvas.draw_idle()

    f2 = ax[0, 1].imshow(seg.edges, cmap='gray')
    ax[0, 1].set_title('Edges')
    ax[0, 1].set_aspect('equal', 'box')

    f3 = ax[1, 0].imshow(seg.clusters, cmap='magma')
    ax[1, 0].set_title("Clusters")
    ax[1, 0].set_aspect('equal', 'box')

    f4 = ax[1, 1].imshow(seg.phases, cmap=cmap)
    ax[1, 1].set_title("Segmented")
    ax[1, 1].set_aspect('equal', 'box')
    get_polygon(seg.clusters, ax[0, 0])

    callback = App(seg, fs=[f1, f2, f3, f4], fig=fig, radio=radio, ax=ax)
    # edge_selector = LassoSelector(ax=ax[0, 1], onselect=callback.newEdges)
    selector = RectangleSelector(ax=ax[0, 0], onselect=callback.onSelect)
    corrector = RectangleSelector(ax=ax[1, 1], onselect=callback.onCorrect)

    # file selection
    bnext = Button(axnext, 'Next Image')
    bprev = Button(axprev, 'Previous Image')

    # threshold selection
    check.on_clicked(callback.check_threshold)

    bnext.on_clicked(callback.next)
    bprev.on_clicked(callback.prev)
    radio.on_clicked(callback.select_phase)
    plt.tight_layout()
    plt.minorticks_on()
    plt.show()
