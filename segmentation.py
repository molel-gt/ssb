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
from matplotlib.widgets import Slider, Button, PolygonSelector, RadioButtons, LassoSelector
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
    "SE": 1,
    "AM": 2,
    }

# defaults
phase = 1
image_id = 0
threshold = 0.0325
NZ = 202
# options
image_ids = np.array(range(NZ))
thresholds = np.linspace(0, 0.1, num=41)
actions = ['label', 'predict']


class SelectFromCollection1:
    """
    Select indices from a matplotlib collection using `PolygonSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.poly = PolygonSelector(ax, self.onselect, draw_bounding_box=True)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

fig, ax = plt.subplots(2, 2, figsize=(12, 12))
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
            neighbors = arr[max(i - dx, 0):min(i + dx, n_max[0] - 1), max(j - dy, 0):min(j + dy, n_max[1] - 1)]
            out[i, j] = np.mean(neighbors)
    return out


def neighborhood_mode(arr, d=(1, 1), n_min=(0, 0), n_max=(501, 501)):
    out = np.zeros(arr.shape)
    dx, dy = d
    for i in range(n_max[0]):
        for j in range(n_max[1]):
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


class Segmentor:
    def __init__(self, image_id=0, threshold=0.0325, input_dir='unsegmented', output_dir='segmentation'):
        self._image_id = image_id
        self._threshold = threshold
        self._output_dir = output_dir
        self._input_dir = input_dir
        self.img = None
        self.clusters = None
        self.edges = None
        self.residual = None
        self.phases = None

    @property
    def image_id(self):
        return self._image_id
    
    @property
    def input_dir(self):
        return self._input_dir
    
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
        img_id = str(image_id).zfill(3)
        if not os.path.exists(os.path.join(self.edges_dir, f'{img_id}')):
        
            img_1 = cv2.imread(os.path.join(self.input_dir, f"{img_id}.tif"), cv2.IMREAD_UNCHANGED)
            img_11 = neighborhood_average(img_1)
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
            img_1 = cv2.imread(os.path.join(self.input_dir, f"{img_id}.tif"), cv2.IMREAD_UNCHANGED)
            with open(os.path.join(self.edges_dir, f'{img_id}'), 'rb') as fp:
                img_2 = pickle.load(fp)

        if not os.path.exists(os.path.join(self.clusters_dir, f'{img_id}')):
            img_3 = neighborhood_average(img_2)
            for i in range(5):
                img_3 = neighborhood_average(img_3)
            img = img_3 / np.max(img_3)
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
            with open(os.path.join(self.clusters_dir, f'{img_id}'), 'wb') as fp:
                pickle.dump(img_cluster_enhanced, fp)
        else:
            with open(os.path.join(self.clusters_dir, f'{img_id}'), 'rb') as fp:
                img_cluster_enhanced = pickle.load(fp)
        
        self.img = img_1
        self.edges = img_2
        self.clusters = img_cluster_enhanced
        print(os.path.exists(os.path.join(self.phases_dir, f'{img_id}')))
        if not os.path.exists(os.path.join(self.phases_dir, f'{img_id}')):
            self.phases = np.zeros(self.img.shape, dtype=np.uint8)
        else:
            with open(os.path.join(self.phases_dir, f'{img_id}'), 'rb') as fp:
                self.phases = pickle.load(fp)

    def run(self, selection=None, phase=None):
        self.create_dirs()
        if self.clusters is None:
            self.clustering()

        if not all([selection, phase]):
            return

        cluster_vals = list(np.unique([self.clusters[iy, ix] for ix, iy in selection]))
        if len(cluster_vals) == 1:
            coords = np.where(self.clusters == cluster_vals[0])
            self.phases[coords] = phase

        with open(os.path.join(self.phases_dir, f'{self.img_id}'), 'wb') as fp:
            pickle.dump(self.phases, fp)
    
    
def run_clustering(image_id=image_id, threshold=threshold, rerun=False, rethreshold=False, action='label'):
    img_id = str(image_id).zfill(3)
    make_dir_if_missing('segmentation')
    make_dir_if_missing('segmentation/edges')
    make_dir_if_missing('segmentation/clusters')
    make_dir_if_missing('segmentation/phases')
    if rerun or rethreshold or not os.path.exists(f'segmentation/edges/{img_id}'):
    
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
        img_1 = cv2.imread(f"unsegmented/{img_id}.tif", cv2.IMREAD_UNCHANGED)
        with open(f'segmentation/edges/{img_id}', 'rb') as fp:
            img_2 = pickle.load(fp)

    if rerun or rethreshold or not os.path.exists(f'segmentation/clusters/{img_id}'):
        img_3 = neighborhood_average(img_2)
        for i in range(5):
            img_3 = neighborhood_average(img_3)
        img = img_3 / np.max(img_3)
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
        with open(f'segmentation/clusters/{img_id}', 'rb') as fp:
            img_cluster_enhanced = pickle.load(fp)

    return img_1, img_2, img_cluster_enhanced


def run_segmentation(image_id, img_1, img_cluster_enhanced):
    image_id = image_id_slider.val
    img_id = str(image_id).zfill(3)
    if not os.path.exists(f"segmentation/phases/{img_id}"):
        img_seg = np.random.randint(0, 2, size=img_1.shape, dtype=np.uint8)
        with open(f'segmentation/phases/{img_id}', 'wb') as fp:
            pickle.dump(img_seg, fp)
    else:
        with open(f'segmentation/phases/{img_id}', 'rb') as fp:
            img_seg = pickle.load(fp)
    return img_seg


def line_picker(line, mouseevent):
    """
    Find the points within a certain distance from the mouseclick in
    data coords and attach some extra attributes, pickx and picky
    which are the data points that were picked.
    """
    if mouseevent.xdata is None:
        return False, dict()
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    maxd = 0.05
    d = np.sqrt(
        (xdata - mouseevent.xdata)**2 + (ydata - mouseevent.ydata)**2)

    ind, = np.nonzero(d <= maxd)
    if len(ind):
        pickx = xdata[ind]
        picky = ydata[ind]
        props = dict(ind=ind, pickx=pickx, picky=picky)
        return True, props
    else:
        return False, dict()


def update_image_id(val):
    image_id = image_id_slider.val
    img_1, img_2, img_cluster_enhanced = run_clustering(image_id=image_id, threshold=threshold, rethreshold=True)
    img_seg = run_segmentation(image_id, img_1, img_cluster_enhanced)

    ax[0, 0].imshow(img_1, cmap='gray')
    ax[0, 0].set_title('Original')
    ax[0, 0].invert_yaxis()
    ax[0, 1].imshow(img_2, cmap='magma')
    ax[0, 1].set_title('Edges')
    ax[0, 1].set_xlim([0, 500])
    ax[0, 1].set_ylim([0, 500])

    coords = np.asarray(np.where(img_cluster_enhanced > -1)).T
    y = np.array([img_cluster_enhanced[ix, iy] for (ix, iy) in coords]).reshape(-1, 1)
    X = np.hstack((coords, y))
    ax[1, 0].scatter(X[:, 1], X[:, 0], cmap=X[:, 2])
    ax[1, 0].set_title("Clusters")
    ax[1, 0].set_xlim([0, 500])
    ax[1, 0].set_ylim([0, 500])

    ax[1, 1].imshow(img_seg, cmap='gray')
    ax[1, 1].set_xlim([0, 500])
    ax[1, 1].set_ylim([0, 500])
    ax[1, 1].set_title("Segmented")
    fig.canvas.draw_idle()


def update_threshold(val):
    image_id = image_id_slider.val
    threshold = threshold_slider.val
    img_1, img_2, img_cluster_enhanced = run_clustering(image_id=image_id, threshold=threshold, rethreshold=True)
    img_seg = run_segmentation(image_id, img_1, img_cluster_enhanced)

    ax[0, 0].imshow(img_1, cmap='gray')
    ax[0, 0].set_title('Original')
    ax[0, 0].invert_yaxis()
    ax[0, 1].imshow(img_2, cmap='magma')
    ax[0, 1].set_title('Edges')
    ax[0, 1].set_xlim([0, 500])
    ax[0, 1].set_ylim([0, 500])

    coords = np.asarray(np.where(img_cluster_enhanced > -1)).T
    y = np.array([img_cluster_enhanced[ix, iy] for (ix, iy) in coords]).reshape(-1, 1)
    X = np.hstack((coords, y))
    ax[1, 0].scatter(X[:, 1], X[:, 0], cmap=X[:, 2])
    ax[1, 0].set_title("Clusters")
    ax[1, 0].set_xlim([0, 500])
    ax[1, 0].set_ylim([0, 500])
    ax[1, 1].imshow(img_seg, cmap='gray')
    ax[1, 1].set_xlim([0, 500])
    ax[1, 1].set_ylim([0, 500])
    ax[1, 1].set_title("Segmented")
    fig.canvas.draw_idle()


def label_clusters(val):
    image_id = image_id_slider.val
    img_id = str(image_id).zfill(3)
    selected_pts = np.array(selector.xys[selector.ind], dtype=int)
    cluster_vals = list(np.unique([img_cluster_enhanced[iy, ix] for ix, iy in selected_pts]))
    print(cluster_vals)
    if len(cluster_vals) == 1:
        coords = np.where(img_cluster_enhanced == cluster_vals[0])
        img_seg[coords] = phase
    with open(f'segmentation/phases/{img_id}', 'wb') as fp:
        pickle.dump(img_seg, fp)
    ax[1, 1].imshow(img_seg, cmap='brg')
    ax[1, 1].set_xlim([0, 500])
    ax[1, 1].set_ylim([0, 500])
    ax[1, 1].set_title("Segmented")
    fig.canvas.draw_idle()
    
    return True


def select_phase(val):
    phase = phases[radio2.value_selected]


def reset(event):
    image_id_slider.reset()
    threshold_slider.reset()


def onpick(event):
    print('onpick2 line:', event.pickx, event.picky)


def accept(event):
        if event.key == "enter":
            print("Selected points:")
            print(selector.xys[selector.ind])
            
            image_id = image_id_slider.val
            img_id = str(image_id).zfill(3)
            selected_pts = np.array(selector.xys[selector.ind], dtype=int)
            cluster_vals = list(np.unique([img_cluster_enhanced[iy, ix] for ix, iy in selected_pts]))
            print(cluster_vals)
            if len(cluster_vals) == 1:
                coords = np.where(img_cluster_enhanced == int(cluster_vals[0]))
                img_seg[coords] = phase
            else:
                for v in cluster_vals:
                    coords = np.where(img_cluster_enhanced == int(v))
                    img_seg[coords] = phase
            with open(f'segmentation/phases/{img_id}', 'wb') as fp:
                pickle.dump(img_seg, fp)
            ax[1, 1].imshow(img_seg, cmap='brg')
            ax[1, 1].set_xlim([0, 500])
            ax[1, 1].set_ylim([0, 500])
            ax[1, 1].set_title("Segmented")
            fig.canvas.draw_idle()

resetax = fig.add_axes([0.935, 0.035, 0.05, 0.025])
image_id_ax = fig.add_axes([0.01, 0.6, 0.025, 0.35])
threshold_ax = fig.add_axes([0.95, 0.6, 0.025, 0.35])
button = Button(resetax, 'Reset', hovercolor='red')
button.on_clicked(reset)

image_id_slider = Slider(
    ax=image_id_ax,
    label='Image',
    orientation='vertical',
    valmin=0,
    valmax=202,
    valinit=image_id,
    valstep=image_ids,
)

threshold_slider = Slider(
    ax=threshold_ax,
    label='Threshold',
    orientation='vertical',
    valmin=0,
    valmax=0.1,
    valinit=threshold,
    valstep=thresholds,
)

##
axcolor = 'lightgoldenrodyellow'
rax = fig.add_axes([0.475, 0.475, 0.05, 0.05], facecolor=axcolor)
radio2 = RadioButtons(rax, ('Void', 'SE', 'AM'), active=2)

radio2.on_clicked(select_phase)

##
image_id = image_id_slider.val
threshold = threshold_slider.val

img_1, img_2, img_cluster_enhanced = run_clustering()
img_seg = run_segmentation(image_id, img_1, img_cluster_enhanced)
ax[0, 0].imshow(img_1, cmap='gray')
ax[0, 0].set_title('Original')

t = ax[0, 0].text(270, 320, "Solid Electrolyte",
            ha="center", va="center", rotation=-35, size=30,
            bbox=dict(boxstyle="darrow,pad=0.1",
                      fc="lightblue", ec="steelblue", lw=2))

ax[0, 0].invert_yaxis()
ax[0, 1].imshow(img_2, cmap='magma')
ax[0, 1].set_title('Edges')
ax[0, 1].set_xlim([0, 500])
ax[0, 1].set_ylim([0, 500])
ax[1, 0].set_box_aspect(1)

coords = np.asarray(np.where(img_cluster_enhanced > -1)).T
y = np.array([img_cluster_enhanced[ix, iy] for (ix, iy) in coords]).reshape(-1, 1)
X = np.hstack((coords, y))
pts = ax[1, 0].scatter(X[:, 1], X[:, 0], c=X[:, 2])
ax[1, 0].set_title("Clusters")
ax[1, 0].set_xlim([0, 500])
ax[1, 0].set_ylim([0, 500])
t = ax[1, 0].text(270, 320, "Solid Electrolyte",
            ha="center", va="center", rotation=-35, size=30,
            bbox=dict(boxstyle="darrow,pad=0.1",
                      fc="lightblue", ec="steelblue", lw=2))
ax[1, 1].imshow(img_seg, cmap='brg')
ax[1, 1].set_xlim([0, 500])
ax[1, 1].set_ylim([0, 500])
ax[1, 1].set_title("Segmented")

selector = SelectFromCollection(ax[1, 0], pts)
fig.canvas.mpl_connect("key_press_event", accept)
image_id_slider.on_changed(update_image_id)
threshold_slider.on_changed(update_threshold)
fig.canvas.draw_idle()
plt.tight_layout()
plt.show()
