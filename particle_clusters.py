#! /usr/bin/env python3

import os

import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from collections import defaultdict
from scipy import linalg

import geometry


def get_neighbors(array_chunk):
    neighbors = np.zeros(array_chunk.shape)

    for idx, val in np.ndenumerate(array_chunk):
        n_neighbors = 0
        if val:
            for counter in range(6):
                new_idx = (idx[0], idx[1], idx[2])
                if counter == 0:
                    # x+1
                    new_idx = (idx[0] + 1, idx[1], idx[2])
                elif counter == 1:
                    # x-1
                    new_idx = (idx[0] - 1, idx[1], idx[2])
                elif counter == 2:
                    # y+1
                    new_idx = (idx[0], idx[1] + 1, idx[2])
                elif counter == 3:
                    # y-1
                    new_idx = (idx[0], idx[1] - 1, idx[2])
                elif counter == 4:
                    # z+1
                    new_idx = (idx[0], idx[1], idx[2] + 1)
                elif counter == 5:
                    # z-1
                    new_idx = (idx[0], idx[1], idx[2] - 1)
                else:
                    raise Exception("Invalid counter")

                try:
                    neigbor_val = array_chunk[tuple(new_idx)]
                except:
                    neigbor_val = 0

                if neigbor_val:
                    n_neighbors += 1
            neighbors[idx] = n_neighbors

    return neighbors


def build_graph(array_chunk):
    """
    :returns: graph
    :rtype: sparse matrix
    """
    G = nx.Graph()
    points = defaultdict(lambda: -1, {})
    valid_points = set([tuple(v) for v in np.argwhere(array_chunk == 1)])
    for idx, value in enumerate(valid_points):
        points[(value[0], value[1], value[2])] = idx
        G.add_node(idx)

    for idx in valid_points:
        old_idx = (idx[0], idx[1], idx[2])

        for counter in range(6):
            new_idx = (idx[0], idx[1], idx[2])
            if counter == 0:
                # x+1
                new_idx = (idx[0] + 1, idx[1], idx[2])
            elif counter == 1:
                # x-1
                new_idx = (idx[0] - 1, idx[1], idx[2])
            elif counter == 2:
                # y+1
                new_idx = (idx[0], idx[1] + 1, idx[2])
            elif counter == 3:
                # y-1
                new_idx = (idx[0], idx[1] - 1, idx[2])
            elif counter == 4:
                # z+1
                new_idx = (idx[0], idx[1], idx[2] + 1)
            elif counter == 5:
                # z-1
                new_idx = (idx[0], idx[1], idx[2] - 1)
            else:
                raise Exception("Invalid counter")

            if new_idx not in valid_points:
                continue
            idx_i = points[old_idx]
            idx_j = points[new_idx]
            if idx_i != idx_j:
                G.add_edge(idx_i, idx_j)

    return points, G


def chunk_array(data, chuck_max_size):
    """
    Split array into chunks
    """
    return


def filter_interior_points(data):
    """
    Masks locations where the voxel has 8 neighbors

    :returns: surface_data
    """
    neighbors = get_neighbors(data)
    surface_data = np.zeros(data.shape)
    with_lt_8_neighbors = np.argwhere(neighbors < 8)
    for idx in with_lt_8_neighbors:
        if data[(idx[0], idx[1], idx[2])] == 1:
            surface_data[(idx[0], idx[1], idx[2])] = 1

    return surface_data


  
class Graph:
  
    # init function to declare class variables
    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]
  
    def DFSUtil(self, temp, v, visited):
  
        # Mark the current vertex as visited
        visited[v] = True
  
        # Store the vertex to list
        temp.append(v)
  
        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:
  
                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp
  
    # method to add an undirected edge
    def addEdge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)
  
    # Method to retrieve connected components
    # in an undirected graph
    def connectedComponents(self):
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.DFSUtil(temp, v, visited))
        return cc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='computes specific area')
    parser.add_argument('--working_dir', help='bmp files directory',
                        required=True)

    args = parser.parse_args()
    working_dir = args.working_dir
    im_files = sorted([os.path.join(working_dir, f) for
                       f in os.listdir(working_dir) if f.endswith(".bmp")])
    n_files = len(im_files)
    data = geometry.load_images_to_logical_array(im_files, x_lims=(0, 10),
                                                 y_lims=(0, 10), z_lims=(0, 10))
    surface_data = filter_interior_points(data)
    points, G = build_graph(surface_data)

    B = nx.adjacency_matrix(G).toarray()
    L = nx.laplacian_matrix(G).toarray()
    L_calc = np.matmul(B, B.transpose())
    cycle_basis = nx.simple_cycles(G)
    ns = linalg.null_space(L)
    for item in nx.connected_components(G):
        print(item)
    print("Number of pieces:", ns.shape[1])
