import itertools
import os


import cv2
import hdbscan
import matplotlib.patches as patches
import matplotlib.pyplot as plt
# %matplotlib ipympl
import mpl_interactions.ipyplot as iplt
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import pyvista
import seaborn as sns
import warnings

from collections import defaultdict
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.ticker import (MultipleLocator)

from PIL import Image
from scipy import ndimage, signal, stats
from skimage import io, filters
from sklearn.ensemble import RandomForestClassifier


# %matplotlib notebook
warnings.simplefilter("ignore")
hdbscan_kwargs = {
    "min_cluster_size": 25,
    "cluster_selection_epsilon": 5,
    "gen_min_span_tree": True
    }
# defaults
image_id = 0
threshold = 0.0325
# options
image_ids = np.array(range(202))
thresholds = np.linspace(0, 0.1, num=41)
actions = ['label', 'predict']
# plotter = pyvista.Plotter()

fig, ax = plt.subplots(3, 2)
plt.subplots_adjust(left=0.15, bottom=0.25)


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
            val = arr[i, j]
            neighbors = arr[max(i - dx, 0):min(i + dx, n_max[0] - 1), max(j - dy, 0):min(j + dy, n_max[1] - 1)]
            out[i, j] = np.mean(neighbors)
    return out


def neighborhood_mode(arr, d=(1, 1), n_min=(0, 0), n_max=(501, 501)):
    out = np.zeros(arr.shape)
    dx, dy = d
    for i in range(n_max[0]):
        for j in range(n_max[1]):
            val = arr[i, j]
            neighbors = arr[max(i - dx, 0):min(i + dx, n_max[0] - 1), max(j - dy, 0):min(j + dy, n_max[1] - 1)]
            out[i, j] = np.mode(neighbors)[0][0]

    return out


def center_of_mass(arr):
    xcom = np.average(arr[:, 0])
    ycom = np.average(arr[:, 1])

    return xcom, ycom


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


def run_clustering(image_id=image_id, threshold=threshold, rerun=False, action='label'):
    img_id = str(image_id).zfill(3)
    make_dir_if_missing('segmentation')
    make_dir_if_missing('segmentation/edges')
    make_dir_if_missing('segmentation/clusters')
    make_dir_if_missing('segmentation/phases')
    if rerun or not os.path.exists(f'segmentation/edges/{img_id}'):
    
        img_1 = cv2.imread(f"unsegmented/{img_id}.tif", cv2.IMREAD_UNCHANGED)
        img_11 = neighborhood_average(img_1)
        for i in range(25):
            img_11 = neighborhood_average(img_11)
        img_2 = filters.meijering(img_11)
        img_3 = neighborhood_average(img_2)
        for i in range(5):
            img_3 = neighborhood_average(img_3)
        img = img_3 / np.max(img_3)
        with open(f'segmentation/edges/{img_id}', 'wb') as fp:
            pickle.dump(img_2, fp)
    else:
        with open(f'segmentation/edges/{img_id}') as fp:
            img_2= pickle.load(fp)

    if rerun or not os.path.exists(f'segmentation/clusters/{img_id}'):
        X_2d = build_features_matrix(img, img_1, threshold)
        y_predict = get_clustering_results(X_2d, **hdbscan_kwargs)
        img_cluster_raw = -2 * np.ones(img.shape)

        for v in np.unique(y_predict):
            X_v = np.where(y_predict == v)[0]
            coords = np.array([X_2d[ix, :2] for ix in X_v])
            for (ix, iy) in coords:
                img_cluster_raw[int(ix), int(iy)] = v

        img_cluster_enhanced = enhance_clusters(img_cluster_raw)
        for i in range(1):
            img_cluster_enhanced = enhance_clusters(img_cluster_raw)
        with open(f'segmentation/clusters/{img_id}', 'wb') as fp:
            pickle.dump(img_cluster_enhanced, fp)
    else:
        with open(f'segmentation/clusters/{img_id}') as fp:
            img_cluster_enhanced = pickle.load(fp)

    return img_1, img_2, img_cluster_enhanced


def run_segmentation(img_1, img_cluster_enhanced):
    img_seg = np.random.randint(0, 2, size=img_1.shape)
    return img_seg


def update_image_id(val):
    image_id = image_id_slider.val
    img_1, img_2, img_cluster_enhanced = run_clustering(image_id=image_id, threshold=threshold)
    img_seg = run_segmentation(img_1, img_cluster_enhanced)
    ax[0, 0].imshow(img_1, cmap='gray')
    ax[0, 0].set_title('Original')
    ax[0, 0].invert_yaxis()
    ax[0, 1].imshow(img_2, cmap='magma')
    ax[0, 1].set_title('Edges')
    ax[0, 1].set_xlim([0, 500])
    ax[0, 1].set_ylim([0, 500])

    ax[1, 0].imshow(img_cluster_enhanced, cmap='magma')
    ax[1, 0].set_title("Enhanced Clusters")
    ax[1, 0].set_xlim([0, 500])
    ax[1, 0].set_ylim([0, 500])

    ax[1, 1].imshow(img_seg, cmap='gray')
    ax[1, 1].set_xlim([0, 500])
    ax[1, 1].set_ylim([0, 500])
    ax[1, 1].set_title("Training Data")


def update_threshold(val):
    threshold = threshold_slider.val
    img_1, img_2, img_cluster_enhanced = run_clustering(image_id=image_id, threshold=threshold)
    img_seg = run_segmentation(img_1, img_cluster_enhanced)

    ax[0, 0].imshow(img_1, cmap='gray')
    ax[0, 0].set_title('Original')
    ax[0, 0].invert_yaxis()
    ax[0, 1].imshow(img_2, cmap='magma')
    ax[0, 1].set_title('Edges')
    ax[0, 1].set_xlim([0, 500])
    ax[0, 1].set_ylim([0, 500])


    ax[1, 0].imshow(img_cluster_enhanced, cmap='magma')
    ax[1, 0].set_title("Enhanced Clusters")
    ax[1, 0].set_xlim([0, 500])
    ax[1, 0].set_ylim([0, 500])
    ax[1, 1].imshow(img_seg, cmap='gray')
    ax[1, 1].set_xlim([0, 500])
    ax[1, 1].set_ylim([0, 500])
    ax[1, 1].set_title("Training Data")
    

def reset(event):
    image_id_slider.reset()
    threshold_slider.reset()



resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')
button.on_clicked(reset)

image_id_slider = Slider(
    ax=ax[2, 0],
    label='Image',
    valmin=0,
    valmax=202,
    valinit=image_id,
    valstep=image_ids,
)

threshold_slider = Slider(
    ax=ax[2, 1],
    label='Threshold',
    valmin=0,
    valmax=0.1,
    valinit=threshold,
    valstep=thresholds,
)
image_id = image_id_slider.val
threshold = threshold_slider.val

img_1, img_2, img_cluster_enhanced = run_clustering()

ax[0, 0].imshow(img_1, cmap='gray')
ax[0, 0].set_title('Original')
ax[0, 0].invert_yaxis()
ax[0, 1].imshow(img_2, cmap='magma')
ax[0, 1].set_title('Edges')
ax[0, 1].set_xlim([0, 500])
ax[0, 1].set_ylim([0, 500])


ax[1, 0].imshow(img_cluster_enhanced, cmap='magma')
ax[1, 0].set_title("Enhanced Clusters")
ax[1, 0].set_xlim([0, 500])
ax[1, 0].set_ylim([0, 500])

ax[1, 1].set_xlim([0, 500])
ax[1, 1].set_ylim([0, 500])
ax[1, 1].set_title("Training Data")

image_id_slider.on_changed(update_image_id)
threshold_slider.on_changed(update_threshold)

fig.canvas.draw()
plt.tight_layout()
plt.show()