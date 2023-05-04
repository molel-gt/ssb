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

        g = ig.Graph(n_nodes + 1, edges)

        self._graph = g

    def get_graph_pieces(self):
        self._pieces = list(self.graph.connected_components())
        self._n_pieces = len(self.pieces)