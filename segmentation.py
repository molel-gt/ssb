#!/usr/bin/env python3
import itertools
import os

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pickle
import warnings

from matplotlib.widgets import Slider, LassoSelector, RadioButtons, TextBox

from skimage import filters
from sklearn.ensemble import RandomForestClassifier


warnings.simplefilter("ignore")


rerun = False
image = None

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

fig, ax = plt.subplots(2, 2)
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
    clusterer = hdbscan.HDBSCAN(**hdbscan_kwargs)
    y_predict = clusterer.fit_predict(X_2d).reshape(-1, 1)

    return y_predict


class Segmentor:
    def __init__(self, image, image_id=0, threshold=0.0325, output_dir='segmentation'):
        self.image_id = image_id
        self.threshold = threshold
        self._output_dir = output_dir
        self.image = image
        self.clusters = None
        self.edges = None
        self.phases = np.zeros(self.image.shape, dtype=np.uint8)
        self.residual = np.zeros(self.image.shape, dtype=np.uint8)
        self.rerun = False

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

    def create_dirs(self):
        make_dir_if_missing(self.output_dir)
        make_dir_if_missing(self.edges_dir)
        make_dir_if_missing(self.clusters_dir)
        make_dir_if_missing(self.phases_dir)
    
    def update_residuals(self):
        coords = np.where(self.phases < 1)
        self.residual[coords] = self.image[coords]
    
    def update_phases(self, selection, phase):
        self.phases[selection] = phase
        self.update_residuals()

        with open(os.path.join(self.phases_dir, f'{str(self.image_id).zfill(3)}'), 'wb') as fp:
            pickle.dump(self.phases, fp)

    def set_edges(self):
        img_11 = neighborhood_average(self.image)
        for i in range(25):
            img_11 = neighborhood_average(img_11)
        img_2 = filters.meijering(img_11)
        self.edges = img_2

    def clustering(self):
        self.set_edges()
        img_3 = neighborhood_average(self.edges)
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
                img_cluster_raw[int(ix), int(iy)] = int(v)

        img_cluster_enhanced = enhance_clusters(img_cluster_raw)

        self.clusters = img_cluster_enhanced

        if os.path.exists(os.path.join(self.phases_dir, f'{str(self.image_id).zfill(3)}')):
            with open(os.path.join(self.phases_dir, f'{str(self.image_id).zfill(3)}'), 'rb') as fp:
                self.phases = pickle.load(fp)
        

    def run(self, selection=None, phase=None, rerun=False, clustering=False, segmentation=False):
        self.rerun = rerun
        self.create_dirs()

        if clustering:
            self.clustering()

        if segmentation:
            self.update_phases(selection, phase)


with open(os.path.join('unsegmented', str(10).zfill(3) + '.tif'), 'rb') as fp:
    image = plt.imread(fp)

seg = Segmentor(image, image_id=10, threshold=0.0325)
seg.run(rerun=False, clustering=True)


def onSelect(val):
    selected_pts = np.array(val, dtype=int)
    cluster_vals = [int(v) for v in np.unique([seg.clusters[iy, ix] for ix, iy in selected_pts]) if v > -1]

    for v in cluster_vals:
        coords = np.where(seg.clusters == v)
        seg.run(selection=coords, phase=phases[radio.value_selected], segmentation=True)

        f4.set_data(seg.phases)
        fig.canvas.draw_idle()


def switch_threshold(val):
    seg.threshold = threshold_slider.val
    seg.run(rerun=False, clustering=True)
    
    f2.set_data(seg.edges)
    f3.set_data(seg.clusters)
    fig.canvas.draw_idle()


def switch_images(val):
    with open(os.path.join('unsegmented', str(img_id_input.text).zfill(3) + '.tif'), 'rb') as fp:
        image = plt.imread(fp)

    seg.image = image
    seg.image_id = int(img_id_input.text)
    seg.threshold = threshold_slider.val
    seg.run(rerun=False, clustering=True)
    f1.set_data(image)
    f2.set_data(seg.edges)
    f3.set_data(seg.clusters)
    f4.set_data(seg.phases)
    fig.canvas.draw_idle()


axcolor = 'lightgoldenrodyellow'
rax = fig.add_axes([0.45, 0.8, 0.1, 0.1], facecolor=axcolor)
img_id_ax = fig.add_axes([0.525, 0.925, 0.025, 0.025])
threshold_ax = fig.add_axes([0.475, 0.4, 0.025, 0.25])
img_id_input = TextBox(img_id_ax, "Image Number :  ", textalignment="right", initial=10)
img_id_input.on_text_change(switch_images)

threshold_slider = Slider(
    ax=threshold_ax,
    label='Threshold',
    orientation='vertical',
    valmin=0,
    valmax=0.2,
    valinit=0.0325,
    valstep=np.linspace(0, 0.2, num=201),
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

f1 = ax[0, 0].imshow(image, cmap='gray')
ax[0, 0].set_title('Original')
ax[0, 0].set_aspect('equal', 'box')
fig.canvas.draw_idle()

f2 = ax[0, 1].imshow(seg.edges, cmap='magma')
ax[0, 1].set_title('Edges')
ax[0, 1].set_box_aspect(1)

f3 = ax[1, 0].imshow(seg.clusters, cmap='magma')
ax[1, 0].set_title("Clusters")
ax[1, 0].set_aspect('equal', 'box')

f4 = ax[1, 1].imshow(seg.phases, cmap='brg')
ax[1, 1].set_title("Segmented")
ax[1, 1].set_aspect('equal', 'box')

selector = LassoSelector(ax=ax[0, 0], onselect=onSelect)
threshold_slider.on_changed(switch_threshold)

fig.canvas.draw_idle()
plt.tight_layout()
plt.show()
