#!/usr/bin/env python3
import itertools
import os

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pickle
import warnings

from ipywidgets import widgets, interactive
from matplotlib.widgets import CheckButtons, Button, Slider, LassoSelector, RadioButtons, TextBox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from skimage import filters
from sklearn.ensemble import RandomForestClassifier


warnings.simplefilter("ignore")


rerun = False
image = None

hdbscan_kwargs = {
    "min_cluster_size": 100,
    "cluster_selection_epsilon": 5,
    "gen_min_span_tree": True
    }

phases = {
    "Void": 0,
    "Solid Electrolyte": 1,
    "Active Material": 2,
    }

training_images = np.linspace(0, 200, num=41)
thresholds = ['-0.99', '-0.95', '-0.80', '-0.03', '-0.02', '0.00', '0.02', '0.03', '0.05', '0.10', '0.20', '0.50', '0.80', '0.95', '0.99']



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


def get_clustering_results(X_2d, **hdbscan_kwargs):
    clusterer = hdbscan.HDBSCAN(**hdbscan_kwargs)
    y_predict = clusterer.fit_predict(X_2d).reshape(-1, 1)

    return y_predict


class Segmentor:
    def __init__(self, image, image_id=0, threshold=0.03, output_dir='segmentation'):
        self.image_id = image_id
        self.threshold = threshold
        self._output_dir = output_dir
        self.image = image
        self._clusters = -2 * np.ones(image.shape) 
        self.edges = None
        self._phases = -1 * np.ones(self.image.shape, dtype=np.intc)
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
        coords = np.where(self.phases < 0)
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
            for i in range(25):
                img_11 = neighborhood_average(img_11)
            img_2 = filters.meijering(img_11)
            self.edges = img_2
            with open(os.path.join(self.edges_dir, f'{str(self.image_id).zfill(3)}'), 'wb') as fp:
                pickle.dump(self.edges, fp)

    def clustering(self):
        if os.path.exists(os.path.join(self.phases_dir, f'{str(self.image_id).zfill(3)}')):
            with open(os.path.join(self.phases_dir, f'{str(self.image_id).zfill(3)}'), 'rb') as fp:
                self._phases = pickle.load(fp)
        else:
            self._phases = -1 * np.ones(self.image.shape, dtype=np.intc)

        self.set_edges()
        img_3 = neighborhood_average(self.edges)
        for i in range(5):
            img_3 = neighborhood_average(img_3)
        img = img_3 / np.max(img_3)
        if self.use_residuals:
            coords = np.where(self.phases > -1)
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
    def __init__(self, seg, selected_phase=1, fs=None, fig=None):
        self.seg = seg
        self.ind = 0
        self._selected_phase = selected_phase
        self._threshold_index = 7
        self._fs = fs
        self._fig = fig
    
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
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def onSelect(self, val):
        selected_pts = np.array(val, dtype=int)
        cluster_vals = [int(v) for v in np.unique([self.seg.clusters[iy, ix] for ix, iy in selected_pts]) if v > -1]
        f1, f2, f3, f4 = self._fs
        for v in cluster_vals:
            coords = np.where(self.seg.clusters == v)
            self.seg.run(selection=coords, phase=self.selected_phase, segmentation=True)
    
            f3.set_data(self.seg.clusters)      
            f4.set_data(self.seg.phases)
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
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
    
    def select_phase(self, val):
        print(val)
        self._selected_phase = phases[radio.value_selected]


class StackSegmentation:
    def __init__(self, training_images, testing_images=None):
        self._model = RandomForestClassifier()
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None
        self._X_validate = None
        self._y_validate = None
        self._training_images = training_images
        self._data = None

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
    def y_validate(self):
        return self._y_validate
    
    @property
    def data(self):
        return self._data

    @property
    def training_images(self):
        return self._training_images

    def build_features_matrix(self):
        self._data = np.zeros((501 * 501 * len(self._training_images), 5), dtype=np.intc)
        train_data = np.zeros((0, 5), dtype=np.intc)
        test_data = np.zeros((0, 5), dtype=np.intc)

        for img_no in self.training_images:
            raw_img = plt.imread(f'unsegmented/{str(int(img_no)).zfill(3)}.tif')
            with open(f'segmentation/phases/{str(int(img_no)).zfill(3)}', 'rb') as fp:
                image = pickle.load(fp)
                coords = np.array(np.where(image > -1), dtype=np.intc).T
                for (ix, iy) in coords:
                    coord = (ix, iy)
                    p = image[coord]

                    row = np.array((int(coord[0]), int(coord[1]), int(img_no), raw_img[int(coord[0]), int(coord[1])], p)).reshape(1, 5)
                if int(int(img_no) % 10) == 0:
                    train_data = np.vstack((train_data, row))
                else:
                    test_data = np.vstack((test_data, row))
        self._X_train = train_data[:, :4]
        self._y_train = train_data[:, 4]
        self._X_test = test_data[:, :4]
        self._y_test = test_data[:, 4]

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        print("Training Score:", self.model.score(self.X_train, self.y_train))

    def validate(self):
        self._y_validate = self.model.predict(self.X_validate)
    
    def test(self):
        self._y_test = self.model.predict(self.X_test)
        print("Testing Score:", self.model.score(self.X_test, self.y_test))

        


if __name__ == '__main__':
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

    seg = Segmentor(image, image_id=image_id, threshold=float(thresholds[7]))
    seg.run(rerun=False, clustering=True)
    fig.suptitle(f"Image: unsegmented/{str(image_id).zfill(3)}.tif")

    axcolor = 'lightgoldenrodyellow'
    rax = inset_axes(ax[0, 2], width="100%", height='70%', loc=3)
    rax.set_facecolor(axcolor)

    # checkbuttons
    check = CheckButtons(ax[1, 2], thresholds, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    # next and previous buttons
    axprev = inset_axes(ax[0, 2], width="49.5%", height='10%', loc=2)
    axnext = inset_axes(ax[0, 2], width="49.5%", height='10%', loc=1)

    radio = RadioButtons(
        rax,
        ('Void', 'Solid Electrolyte', 'Active Material'),
        active=1,
        # label_props={'color': ['blue', 'red' , 'green']},
        # radio_props={'edgecolor': ['darkblue', 'darkred', 'darkgreen'],
        #               'facecolor': ['blue', 'red', 'green'],
        #               },
        )

    f1 = ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].set_title('Original')
    ax[0, 0].set_aspect('equal', 'box')
    fig.canvas.draw_idle()

    f2 = ax[0, 1].imshow(seg.edges, cmap='gray')
    ax[0, 1].set_title('Edges')
    ax[0, 1].set_aspect('equal', 'box')

    f3 = ax[1, 0].imshow(seg.clusters, cmap='magma')
    ax[1, 0].set_title("Clusters")
    ax[1, 0].set_aspect('equal', 'box')

    f4 = ax[1, 1].imshow(seg.phases, cmap='brg')
    ax[1, 1].set_title("Segmented")
    ax[1, 1].set_aspect('equal', 'box')

    callback = App(seg, fs=[f1, f2, f3, f4], fig=fig)
    selector = LassoSelector(ax=ax[0, 0], onselect=callback.onSelect)

    # file selection
    bnext = Button(axnext, 'Next Image')
    bprev = Button(axprev, 'Previous Image')

    # threshold selection
    check.on_clicked(callback.check_threshold)

    bnext.on_clicked(callback.next)
    bprev.on_clicked(callback.prev)
    radio.on_clicked(callback.select_phase)
    plt.tight_layout()
    plt.show()
