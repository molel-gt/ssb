#!/usr/bin/env python3
import argparse
import os

import numpy as np
import commons


markers = commons.Markers()


def read_nodes_file(nodes_file_path):
    """"""
    nodes = {}
    with open(nodes_file_path) as fp:
        rows = fp.readlines()
        for row in rows[1:]:
            parts = row.strip("\n").split()
            nodes[int(parts[0])] = tuple([float(v) for v in parts[1:4]])
    return nodes


def read_faces_file(faces_file_path):
    """"""
    faces = []
    with open(faces_file_path) as fp:
        rows = fp.readlines()
        for row in rows[1:]:
            parts = tuple([int(v) for v in row.strip("\n").split(" ")])
            faces.append(parts)
    return faces


def read_tets_file(tets_file_path, label):
    """"""
    tets = {}
    with open(tets_file_path) as fp:
        rows = fp.readlines()
        for row in rows[1:]:
            parts = tuple([int(v) for v in row.strip("\n").split()])
            tets[int(parts[0])] = tuple([int(v) for v in parts[1:5]] + [label])
    return tets


def merge_nodes(nodes_1, nodes_2):
    nodes_dict = {}
    nodes_dict.update(nodes_1)
    nodes_2_lookup = {}
    idx = np.max(list(nodes_dict.keys())) + 1

    idx2 = 1
    for k, node in nodes_2.items():
        val = nodes_dict.get(node)
        if val is None:
            nodes_dict[idx] = node
            nodes_2_lookup[int(k)] = int(idx)
            idx += 1
        else:
            nodes_2_lookup[int(k)] = int(val)

    return nodes_dict, nodes_2_lookup


def translate_tets(input_tets, nodes_2_lookup, shift_idx=0):
    output_tets = {}
    for idx, tet in input_tets.items():
        new_tet = [nodes_2_lookup.get(t, t) for t in tet[:4]] + list(tet[4:])
        output_tets[int(idx+shift_idx)] = new_tet
    return output_tets


def write_nodes_to_file(nodes, nodes_file_path, scale):
    sx, sy, sz = scale
    with open(nodes_file_path, "w") as fp:
        fp.write(f"{len(nodes.keys())} 3\n")
        for idx, (x, y, z) in nodes.items():
            fp.write(f"{idx} {x*sx} {y*sy} {z*sz}\n")
    return


def write_tets_to_file(tets, tets_file_path):
    with open(tets_file_path, "w") as fp:
        fp.write(f"{len(tets.keys())} 4 1\n")
        for idx, (p0, p1, p2, p3, label) in tets.items():
            fp.write(f"{idx} {p0} {p1} {p2} {p3} {label}\n")
    return


if __name__ == '__main__':
    workdir = os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/composite-electrode/")
    sse_nodes_file = os.path.join(workdir, "2-raw.1.node")
    cam_nodes_file = os.path.join(workdir, "3-raw.1.node")
    merged_nodes_file = os.path.join(workdir, "merged.node")

    sse_tets_file = os.path.join(workdir, "2-raw.1.ele")
    cam_tets_file = os.path.join(workdir, "3-raw.1.ele")
    merged_tets_file = os.path.join(workdir, "merged.ele")

    # reads nodes
    nodes_1 = read_nodes_file(sse_nodes_file)
    nodes_2 = read_nodes_file(cam_nodes_file)

    tets_1 = read_tets_file(sse_tets_file, markers.electrolyte)
    tets_2 = read_tets_file(cam_tets_file, markers.positive_am)

    nodes_dict, nodes_2_lookup = merge_nodes(nodes_1, nodes_2)

    tets_2_new = translate_tets(tets_2, nodes_2_lookup, shift_idx=np.max(list(tets_1.keys())))
    tets_dict = {}
    tets_dict.update(tets_1)
    tets_dict.update(tets_2_new)

    scale = [0.08e-06/(1000*0.08e-06), 0.08e-06/(1000*0.08e-06), 0.2e-06/(1000*0.08e-06)]
    write_nodes_to_file(nodes_dict, merged_nodes_file, scale)
    write_tets_to_file(tets_dict, merged_tets_file)