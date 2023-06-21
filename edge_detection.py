#!/usr/bin/env python
# coding: utf-8

import json
import os

import alphashape
import argparse

from collections import defaultdict
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy

import warnings

from skimage import filters

import segmentation as seg, grapher, utils


warnings.simplefilter("ignore")

levels = {
    0: ('less_than', 0.2, lambda size: size >= 0.05),
    1: ('greater_than', 0.05, lambda size: False),
}

NX = NY = 501


def coord2idx(x, y, NY=NY):
    return x * (NY - 1) + y


def idx2coord(idx, NY=NY):
    return int(idx / (NY - 1)), int(idx % (NY - 1))


def points_inside_polygon(polygon, points_arr):
    path = Path(polygon)
    containing = path.contains_points(points_arr)

    return points_arr[containing]


def write_edges_to_file(edges, img_id, outdir):
    with open(os.path.join(outdir, f'{str(img_id).zfill(3)}.json'), 'w') as f:
        json.dump(edges, f)


def get_raw_clusters(img, img_edges, condition='less_than', threshold=0.2):
    if condition not in ['less_than', 'greater_than']:
        raise ValueError(f'Unsupported condition {condition}')
    if condition == 'less_than':
        features = seg.build_features_matrix(img_edges < threshold, img, 0.05)
    else:
        features = seg.build_features_matrix(img_edges > threshold, img, 0.05)
    clusters_0 = seg.get_clustering_results(features[:, :3], **seg.hdbscan_kwargs)
    clusters = -1 * np.ones(img.shape, dtype=np.int32)
    for i in range(features.shape[0]):
        x, y = [int(v) for v in features[i, :2]]
        clusters[x, y] = clusters_0[i]
    
    return clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstructs volume from segemented images.')
    parser.add_argument('--img_id', help='image id', type=int, required=True)
    parser.add_argument('--indir', help='input directory', type=str, nargs='?', const=1, default='unsegmented')
    parser.add_argument('--outdir', help='output directory', type=str, nargs='?', const=1, default='segmentation')
    args = parser.parse_args()
    img = np.asarray(plt.imread(os.path.join(args.indir, f'{str(args.img_id).zfill(3)}.tif')))
    NX, NY = img.shape
    utils.make_dir_if_missing(args.outdir)
    edges_dir = os.path.join(args.outdir, 'edges')
    utils.make_dir_if_missing(edges_dir)

    img_01 = seg.neighborhood_average(img, d=(5, 5))
    img_1 = filters.meijering(img_01)

    img_trial = img * (1 - img_1 / np.max(img_1))
    img_trial_edges = filters.meijering(img_trial)
    points = {}
    points_arr = np.zeros((NX * NY, 2), dtype=np.int32)
    counter = 0
    for ix in range(NX):
        for iy in range(NY):
            idx = coord2idx(ix, iy, NY=NY)
            points[ix, iy] = idx
            points_arr[idx, :] = (ix, iy)

    edges = defaultdict(list)

    for level, (condition, threshold, size_check) in levels.items():
        clusters = get_raw_clusters(img_trial, img_trial_edges, condition=condition, threshold=threshold)
        for v in np.unique(clusters):
            if v < 0:
                continue
            coords = np.where(np.isclose(clusters, v))
            if coords[0].shape[0] < 5:
                    continue
            size = coords[0].shape[0] / 500 ** 2
            if size <= 0:
                continue
            arr = []
            for idx in range(coords[0].shape[0]):
                c = (coords[1][idx], coords[0][idx])
                arr.append(c)
            points_view = {}
            for i in range(len(arr)):
                coord = arr[i]
                idx = int(points[coord])
                points_view[idx] = coord
            graph = grapher.PixelGraph(points=points_view)
            graph.build_graph()
            graph.get_graph_pieces()
            if size_check(size):
                continue
            if size < 0.05:
                hull = []
                try:
                    alpha_shape = alphashape.alphashape(arr, 0.2)
                    exterior = alpha_shape.exterior
                    for c in exterior.coords:
                        hull.append((c[0], c[1]))
                except scipy.spatial._qhull.QhullError as e1:
                    pass
                except AttributeError as e2:
                    pass
                if len(hull) == 0:
                    continue
                hull_ids = []
                for i in range(len(hull)):
                    idx = points[hull[i]]
                    hull_ids.append(idx)
                edges[level].append(hull_ids)

            else:
                for p in graph.pieces:
                    if len(p) < 5:
                        continue
                    arr2 = []
                    for c in p:
                        arr2.append(points_view[int(c)])
                    hull = []
                    try:
                        alpha_shape = alphashape.alphashape(arr2, 0.2)
                        exterior = alpha_shape.exterior
                        for c in exterior.coords:
                            hull.append((c[0], c[1]))
                    except scipy.spatial._qhull.QhullError as e1:
                        pass
                    except AttributeError as e2:
                        pass
                    if len(hull) == 0:
                        continue
                    hull_ids = []
                    for i in range(len(hull)):
                        idx = points[hull[i]]
                        hull_ids.append(idx)
                    edges[level].append(hull_ids)

    write_edges_to_file(edges, args.img_id, edges_dir)

    print(points_inside_polygon(hull, points_arr))
