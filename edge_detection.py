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

levels1 = {
    0: ('less_than', 0.2, lambda size: size >= 0.05),
    1: ('greater_than', 0.05, lambda size: False),
}

levels2 = {
    2: ('less_than', 0.1, lambda size: size >= 0.05),
}

NX = NY = 501


def coord2idx(x, y, NY=NY):
    return int(x * (NY - 1) + y)


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


def get_edges(img_input, img_edges, levels):
    NX, NY = img_input.shape
    edges_ = defaultdict(list)
    for level, (condition, threshold, size_check) in levels.items():
        clusters = get_raw_clusters(img_input, img_edges, condition=condition, threshold=threshold)
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
                idx = coord2idx(*coord, NY=NY)
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
                    idx = coord2idx(*hull[i], NY=NY)
                    hull_ids.append(idx)
                edges_[int(level)].append(hull_ids)

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
                        idx = coord2idx(*hull[i], NY=NY)
                        hull_ids.append(idx)
                    edges_[int(level)].append(hull_ids)

    return edges_


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

    img_input = img * (1 - img_1 / np.max(img_1))
    img_edges = filters.meijering(img_input)
    points_arr = np.zeros((NX * NY, 2), dtype=np.int32)
    counter = 0
    for ix in range(NX):
        for iy in range(NY):
            idx = coord2idx(ix, iy, NY=NY)
            points_arr[idx, :] = (ix, iy)

    edges1 = get_edges(img_input, img_edges, levels1)

    # process edges 1
    img_res = img_edges.copy()
    for level in ['0', '1']:
        level_edges = edges1[level]
        for point_set in level_edges:
            polygon = []
            for idx in point_set:
                x, y = idx2coord(idx)
                polygon.append((x, y))
            #         if len(polygon) < 25:
            #             continue
            inside_points = points_inside_polygon(polygon, points_arr)
            if inside_points.shape[0] == 0:
                continue
            if inside_points.shape[0] == 1:
                img_res[inside_points[0, 0], inside_points[0, 1]] = 1
            else:
                arr = np.asarray(inside_points)
                if arr.shape[0] > 1000 and level == '0':
                    continue
                if level == '1' and arr.shape[0] > 68000:
                    continue
                aspect = arr.shape[0] / len(polygon)
                aspect2 = 4 * arr.shape[0] / len(polygon) ** 2
                if aspect > 40 and len(polygon) < 500:
                    print(aspect, aspect2, len(polygon))
                    continue
                elif 27 < aspect < 28:
                    print(aspect, aspect2, len(polygon))
                    continue
                else:
                    if 100 < aspect < 110:
                        print(aspect, len(polygon))
                    img_res[(arr[:, 0], arr[:, 1])] = 1
    edges2 = get_edges(img_input, img_res, levels2)

    edges_final = {}
    for k, v in edges1.items():
        if len(v) == 0:
            continue
        edges_final[int(k)] = v
    for k, v in edges2.items():
        if len(v) == 0:
            continue
        edges_final[int(k)] = v

    write_edges_to_file(edges_final, args.img_id, edges_dir)
