#!/usr/bin/env python3
import itertools
import os

import cv2
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pickle
import warnings

from matplotlib.image import AxesImage
from matplotlib.widgets import Slider, LassoSelector, RadioButtons, TextBox
from matplotlib.path import Path

from skimage import filters
from sklearn.ensemble import RandomForestClassifier


warnings.simplefilter("ignore")


hdbscan_kwargs = {
    "min_cluster_size": 25,
    "cluster_selection_epsilon": 5,
    "gen_min_span_tree": True
    }

phases = {
    "Void": 0,
    "Solid Electrolyte": 1,
    "Active Material": 2,
    }

fig, ax = plt.subplots(4, 2)
# plt.subplots_adjust(left=0.25, bottom=0.25)
ax[0, 0].grid(which='both')
ax[1, 0].grid(which='both')
ax[0, 1].grid(which='both')
ax[1, 1].grid(which='both')


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
    coords = np.asarray(np.where(np.logical_and(np.greater_equal(img, 0), np.less_equal(img, threshold)))).T
    y = np.array([img_1[ix, iy] for (ix, iy) in coords]).reshape(-1, 1) / 255
    y_ = np.array([img[ix, iy] for (ix, iy) in coords]).reshape(-1, 1)
    X_2d = np.hstack((coords, y, y_))

    return X_2d


def get_clustering_results(X_2d, **hdbscan_kwargs):
    # hdbscan
    clusterer = hdbscan.HDBSCAN(**hdbscan_kwargs)
    y_predict = clusterer.fit_predict(X_2d).reshape(-1, 1)

    return y_predict


class Segmentor:
    def __init__(self, image, image_id=0, threshold=0.0325, output_dir='segmentation'):
        self._image_id = image_id
        self._threshold = threshold
        self._output_dir = output_dir
        self.img = image
        self.clusters = None
        self.edges = None
        self.residual = None
        self.phases = None

    @property
    def image_id(self):
        return self._image_id

    @property
    def output_dir(self):
        return self._output_dir

    @property
    def edges_dir(self):
        return os.path.join(self.output_dir, 'edges')

    @property
    def clusters_dir(self):
        return os.path.join(self.output_dir, 'clusters')
    
    @property
    def phases_dir(self):
        return os.path.join(self.output_dir, 'phases')

    @property
    def threshold(self):
        return self._threshold
    
    def create_dirs(self):
        make_dir_if_missing(self.output_dir)
        make_dir_if_missing(self.edges_dir)
        make_dir_if_missing(self.clusters_dir)
        make_dir_if_missing(self.phases_dir)

    def clustering(self):
        img_id = str(self.image_id).zfill(3)
        if not os.path.exists(os.path.join(self.edges_dir, f'{img_id}')):
        
            img_11 = neighborhood_average(self.image)
            for i in range(25):
                img_11 = neighborhood_average(img_11)
            img_2 = filters.meijering(img_11)
            img_3 = neighborhood_average(img_2)
            for i in range(5):
                img_3 = neighborhood_average(img_3)
            img = img_3 / np.max(img_3)
            with open(os.path.join(self.edges_dir, f'{img_id}'), 'wb') as fp:
                pickle.dump(img_2, fp)
        else:
            with open(os.path.join(self.edges_dir, f'{img_id}'), 'rb') as fp:
                img_2 = pickle.load(fp)
        self.edges = img_2

        img_3 = neighborhood_average(img_2)
        for i in range(5):
            img_3 = neighborhood_average(img_3)
        img = img_3 / np.max(img_3)
        X_2d = build_features_matrix(img, self.image, self.threshold)
        y_predict = get_clustering_results(X_2d, **hdbscan_kwargs)
        img_cluster_raw = -2 * np.ones(img.shape)  # -2, -1 are residual non-clustered

        for v in np.unique(y_predict):
            X_v = np.where(y_predict == v)[0]
            coords = np.array([X_2d[ix, :2] for ix in X_v])
            for (ix, iy) in coords:
                img_cluster_raw[int(ix), int(iy)] = v

        img_cluster_enhanced = enhance_clusters(img_cluster_raw)
        for i in range(1):
            img_cluster_enhanced = enhance_clusters(img_cluster_raw)
        with open(os.path.join(self.clusters_dir, f'{img_id}'), 'wb') as fp:
            pickle.dump(img_cluster_enhanced, fp)


        self.clusters = img_cluster_enhanced

        if not os.path.exists(os.path.join(self.phases_dir, f'{img_id}')):
            self.phases = np.zeros(self.img.shape, dtype=np.uint8)
        else:
            with open(os.path.join(self.phases_dir, f'{img_id}'), 'rb') as fp:
                self.phases = pickle.load(fp)
        self.residual = self.image[np.where(self.phases < 1)]

    def run(self, selection=None, phase=None):
        img_id = str(self.image_id).zfill(3)
        self.create_dirs()
        if self.clusters is None:
            self.clustering()

        if not all([selection, phase]):
            return

        self.phases[selection] = phase

        img_id = str(self.image_id).zfill(3)

        with open(os.path.join(self.phases_dir, f'{img_id}'), 'wb') as fp:
            pickle.dump(self.phases, fp)


def update_view(val):
    seg = Segmentor(image, image_id=int(img_id_input.text), threshold=threshold_slider.val)
    seg.run()
    img = seg.img
    edges = seg.edges
    clusters = seg.clusters
    img_seg = seg.phases
    
    f1.set_data(img)
    f2.set_data(edges)
    f3.set_data(clusters)
    f4.set_data(img_seg)

    fig.canvas.draw_idle()


def onSelect(val):
    selected_pts = np.array(val, dtype=int)

    seg = Segmentor(image, image_id=int(img_id_input.text), threshold=threshold_slider.val)
    seg.run()
    img = seg.img
    edges = seg.edges
    clusters = seg.clusters
    img_seg = seg.phases

    cluster_vals = [int(v) for v in np.unique([clusters[iy, ix] for ix, iy in selected_pts]) if v > -1]

    for v in cluster_vals:
        coords = np.where(clusters == v)
        seg.run(selection=coords, phase=phases[radio.value_selected])

    img = seg.img
    edges = seg.edges
    clusters = seg.clusters
    img_seg = seg.phases

    f1.set_data(img)
    f2.set_data(edges)
    f3.set_data(clusters)
    f4.set_data(img_seg)

    fig.canvas.draw_idle()


axcolor = 'lightgoldenrodyellow'
rax = fig.add_axes([0.45, 0.8, 0.1, 0.1], facecolor=axcolor)
img_id_ax = fig.add_axes([0.525, 0.925, 0.025, 0.025])
threshold_ax = fig.add_axes([0.475, 0.6, 0.025, 0.15])
img_id_input = TextBox(img_id_ax, "Image Number :  ", textalignment="right", initial=0)
img_id_input.on_text_change(update_view)

threshold_slider = Slider(
    ax=threshold_ax,
    label='Threshold',
    orientation='vertical',
    valmin=0,
    valmax=0.1,
    valinit=0.0325,
    valstep=[0, 0.01, 0.02, 0.03, 0.0325, 0.04, 0.05, 0.075, 0.1],
)

radio = RadioButtons(
    rax,
    ('Void', 'Solid Electrolyte', 'Active Material'),
    active=0,
    label_props={'color': ['blue', 'red' , 'green']},
    radio_props={'edgecolor': ['darkblue', 'darkred', 'darkgreen'],
                 'facecolor': ['blue', 'red', 'green'],
                 },
    )

seg = Segmentor(image_id=int(img_id_input.text), threshold=threshold_slider.val)
seg.run()
img = seg.img
edges = seg.edges
clusters = seg.clusters
img_seg = seg.phases

f1 = ax[0, 0].imshow(img, cmap='gray')
ax[0, 0].set_title('Original')
ax[0, 0].set_aspect('equal', 'box')

f2 = ax[0, 1].imshow(edges, cmap='magma')
ax[0, 1].set_title('Edges')
ax[0, 1].set_box_aspect(1)

f3 = ax[1, 0].imshow(clusters, cmap='magma')
ax[1, 0].set_title("Clusters")
ax[1, 0].set_aspect('equal', 'box')

f4 = ax[1, 1].imshow(img_seg, cmap='brg')
ax[1, 1].set_title("Segmented")
ax[1, 1].set_aspect('equal', 'box')

selector = LassoSelector(ax=ax[0, 0], onselect=onSelect)
threshold_slider.on_changed(update_view)

fig.canvas.draw_idle()
plt.tight_layout()
plt.show()
