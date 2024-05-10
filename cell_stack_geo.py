#!/usr/bin/env python3
import argparse
import os

import gmsh
import numpy as np


class Boundaries:
    def __init__(self):
        pass

    @property
    def negative_cc_1_graphite_1(self):
        return 1

    @property
    def graphite_1_separator_1(self):
        return 2

    @property
    def separator_1_nmc_1(self):
        return 3

    @property
    def nmc_1_positive_cc_1(self):
        return 4

    @property
    def positive_cc_1_nmc_2(self):
        return 5

    @property
    def nmc_2_separator_2(self):
        return 6

    @property
    def separator_2_graphite_2(self):
        return 7

    @property
    def graphite_2_negative_cc_2(self):
        return 8

    @property
    def bottom(self):
        return 9

    @property
    def top(self):
        return 10

    @property
    def insulated_negative_cc_1(self):
        return 11

    @property
    def insulated_graphite_1(self):
        return 12

    @property
    def insulated_separator_1(self):
        return 13

    @property
    def insulated_nmc_1(self):
        return 14

    @property
    def insulated_positive_cc_1(self):
        return 15

    @property
    def insulated_nmc_2(self):
        return 16

    @property
    def insulated_separator_2(self):
        return 17

    @property
    def insulated_graphite_2(self):
        return 18

    @property
    def insulated_negative_cc_2(self):
        return 19

    @property
    def insulated_free_electrolyte(self):
        return 20

    @property
    def insulated(self):
        return 21

    @property
    def positive_tab_1(self):
        return 22

    @property
    def negative_tab_1(self):
        return 23

    @property
    def negative_tab_2(self):
        return 24


class Phases:
    def __init__(self):
        pass

    @property
    def negative_cc_1(self):
        return 1

    @property
    def graphite_1(self):
        return 2

    @property
    def separator_1(self):
        return 3

    @property
    def nmc_1(self):
        return 4

    @property
    def positive_cc_1(self):
        return 5

    @property
    def nmc_2(self):
        return 6

    @property
    def separator_2(self):
        return 7

    @property
    def graphite_2(self):
        return 8

    @property
    def negative_cc_2(self):
        return 9

    @property
    def free_electrolyte(self):
        return 10


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lithium plating')
    parser.add_argument('--resolution', help=f'max resolution resolution (microns)', nargs='?', const=1, default=1, type=float)
    args = parser.parse_args()
    micron = 1e-6
    resolution = args.resolution * micron
    boundaries = Boundaries()
    phases = Phases()
    output_meshfile = 'mesh.msh'
    points_left = [
    (0, 0, 0),
    (0, 25 * micron, 0),
    (0, 75 * micron, 0),
    (0, 100 * micron, 0),
    (0, 150 * micron, 0),
    (0, 175 * micron, 0),
    (0, 225 * micron, 0),
    (0, 250 * micron, 0),
    (0, 300 * micron, 0),
    (0, 325 * micron, 0),
    ]
    points_1 = [
    (50 * micron, 0, 0),
    (50 * micron, 25 * micron, 0),
    (50 * micron, 75 * micron, 0),
    (50 * micron, 100 * micron, 0),
    (50 * micron, 150 * micron, 0),
    (50 * micron, 175 * micron, 0),
    (50 * micron, 225 * micron, 0),
    (50 * micron, 250 * micron, 0),
    (50 * micron, 300 * micron, 0),
    (50 * micron, 325 * micron, 0),
    ]
    points_2 = [
    (100 * micron, 0, 0),
    (100 * micron, 25 * micron, 0),
    (100 * micron, 75 * micron, 0),
    (100 * micron, 100 * micron, 0),
    (100 * micron, 150 * micron, 0),
    (100 * micron, 175 * micron, 0),
    (100 * micron, 225 * micron, 0),
    (100 * micron, 250 * micron, 0),
    (100 * micron, 300 * micron, 0),
    (100 * micron, 325 * micron, 0),
    ]
    points_3 = [
    (150 * micron, 0, 0),
    (150 * micron, 25 * micron, 0),
    (150 * micron, 75 * micron, 0),
    (150 * micron, 100 * micron, 0),
    (150 * micron, 150 * micron, 0),
    (150 * micron, 175 * micron, 0),
    (150 * micron, 225 * micron, 0),
    (150 * micron, 250 * micron, 0),
    (150 * micron, 300 * micron, 0),
    (150 * micron, 325 * micron, 0),
    ]
    points_4 = [
    (200 * micron, 0, 0),
    (200 * micron, 25 * micron, 0),
    (200 * micron, 75 * micron, 0),
    (200 * micron, 100 * micron, 0),
    (200 * micron, 150 * micron, 0),
    (200 * micron, 175 * micron, 0),
    (200 * micron, 225 * micron, 0),
    (200 * micron, 250 * micron, 0),
    (200 * micron, 300 * micron, 0),
    (200 * micron, 325 * micron, 0),
    ]
    points_5 = [
    (3100 * micron, 0, 0),
    (3100 * micron, 25 * micron, 0),
    (3100 * micron, 75 * micron, 0),
    (3100 * micron, 100 * micron, 0),
    (3100 * micron, 150 * micron, 0),
    (3100 * micron, 175 * micron, 0),
    (3100 * micron, 225 * micron, 0),
    (3100 * micron, 250 * micron, 0),
    (3100 * micron, 300 * micron, 0),
    (3100 * micron, 325 * micron, 0),
    ]
    points_6 = [
    (3150 * micron, 0, 0),
    (3150 * micron, 25 * micron, 0),
    (3150 * micron, 75 * micron, 0),
    (3150 * micron, 100 * micron, 0),
    (3150 * micron, 150 * micron, 0),
    (3150 * micron, 175 * micron, 0),
    (3150 * micron, 225 * micron, 0),
    (3150 * micron, 250 * micron, 0),
    (3150 * micron, 300 * micron, 0),
    (3150 * micron, 325 * micron, 0),
    ]
    points_7 = [
    (3200 * micron, 0, 0),
    (3200 * micron, 25 * micron, 0),
    (3200 * micron, 75 * micron, 0),
    (3200 * micron, 100 * micron, 0),
    (3200 * micron, 150 * micron, 0),
    (3200 * micron, 175 * micron, 0),
    (3200 * micron, 225 * micron, 0),
    (3200 * micron, 250 * micron, 0),
    (3200 * micron, 300 * micron, 0),
    (3200 * micron, 325 * micron, 0),
    ]
    points_8 = [
    (3250 * micron, 0, 0),
    (3250 * micron, 25 * micron, 0),
    (3250 * micron, 75 * micron, 0),
    (3250 * micron, 100 * micron, 0),
    (3250 * micron, 150 * micron, 0),
    (3250 * micron, 175 * micron, 0),
    (3250 * micron, 225 * micron, 0),
    (3250 * micron, 250 * micron, 0),
    (3250 * micron, 300 * micron, 0),
    (3250 * micron, 325 * micron, 0),
    ]
    points_9 = [
    (3300 * micron, 0, 0),
    (3300 * micron, 25 * micron, 0),
    (3300 * micron, 75 * micron, 0),
    (3300 * micron, 100 * micron, 0),
    (3300 * micron, 150 * micron, 0),
    (3300 * micron, 175 * micron, 0),
    (3300 * micron, 225 * micron, 0),
    (3300 * micron, 250 * micron, 0),
    (3300 * micron, 300 * micron, 0),
    (3300 * micron, 325 * micron, 0),
    ]
    points_right = [
    (3350 * micron, 0, 0),
    (3350 * micron, 25 * micron, 0),
    (3350 * micron, 75 * micron, 0),
    (3350 * micron, 100 * micron, 0),
    (3350 * micron, 150 * micron, 0),
    (3350 * micron, 175 * micron, 0),
    (3350 * micron, 225 * micron, 0),
    (3350 * micron, 250 * micron, 0),
    (3350 * micron, 300 * micron, 0),
    (3350 * micron, 325 * micron, 0),
    ]

    points = np.zeros((10, 11), dtype=np.intc)
    lines_vertical = np.zeros((9, 11), dtype=np.intc)
    lines_horizontal = np.zeros((10, 10), dtype=np.intc)
    loops = np.zeros((9, 10), dtype=np.intc)
    surfaces = np.zeros((9, 10), dtype=np.intc)

    gmsh.initialize()
    gmsh.model.add('cell-stack')

    for idx, p in enumerate(points_left):
        points[idx, 0] = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    for idx, p in enumerate(points_1):
        points[idx, 1] = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    for idx, p in enumerate(points_2):
        points[idx, 2] = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    for idx, p in enumerate(points_3):
        points[idx, 3] = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    for idx, p in enumerate(points_4):
        points[idx, 4] = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    for idx, p in enumerate(points_5):
        points[idx, 5] = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    for idx, p in enumerate(points_6):
        points[idx, 6] = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    for idx, p in enumerate(points_7):
        points[idx, 7] = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    for idx, p in enumerate(points_8):
        points[idx, 8] = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    for idx, p in enumerate(points_9):
        points[idx, 9] = gmsh.model.occ.addPoint(*p, meshSize=resolution)
    for idx, p in enumerate(points_right):
        points[idx, 10] = gmsh.model.occ.addPoint(*p, meshSize=resolution)

    for col in range(11):
        for row in range(9):
            lines_vertical[row, col] = gmsh.model.occ.addLine(points[row, col], points[row + 1, col])

    for row in range(10):
        for col in range(10):
            lines_horizontal[row, col] = gmsh.model.occ.addLine(points[row, col], points[row, col + 1])

    gmsh.model.occ.synchronize()

    for row in range(9):
        for col in range(10):
            loops[row, col] = gmsh.model.occ.addCurveLoop(
                [lines_horizontal[row, col],
                lines_vertical[row, col+1],
                lines_horizontal[row+1, col],
                lines_vertical[row, col]]
                )
            surfaces[row, col] = gmsh.model.occ.addPlaneSurface([loops[row, col]])
            gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(1, list(lines_horizontal[0, 0:-3]), boundaries.bottom, "bottom")
    gmsh.model.addPhysicalGroup(1, list(lines_horizontal[1, 3:-3]), boundaries.negative_cc_1_graphite_1, "negative_cc_1_graphite_1")
    gmsh.model.addPhysicalGroup(1, list(lines_horizontal[2, 3:-3]), boundaries.graphite_1_separator_1, "graphite_1_separator_1")
    gmsh.model.addPhysicalGroup(1, list(lines_horizontal[3, 4:-4]), boundaries.separator_1_nmc_1, "separator_1_nmc_1")
    gmsh.model.addPhysicalGroup(1, list(lines_horizontal[4, 4:-4]), boundaries.nmc_1_positive_cc_1, "nmc_1_positive_cc_1")
    gmsh.model.addPhysicalGroup(1, list(lines_horizontal[5, 4:-4]), boundaries.positive_cc_1_nmc_2, "positive_cc_1_nmc_2")
    gmsh.model.addPhysicalGroup(1, list(lines_horizontal[6, 4:-4]), boundaries.nmc_2_separator_2, "nmc_2_separator_2")
    gmsh.model.addPhysicalGroup(1, list(lines_horizontal[7, 3:-3]), boundaries.separator_2_graphite_2, "separator_2_graphite_2")
    gmsh.model.addPhysicalGroup(1, list(lines_horizontal[-2, 3:-3]), boundaries.graphite_2_negative_cc_2, "graphite_2_negative_cc_2")
    gmsh.model.addPhysicalGroup(1, list(lines_horizontal[-1, 0:-3]), boundaries.top, "top")

    # current tabs
    gmsh.model.addPhysicalGroup(1, list(lines_vertical[4, [-2]]), boundaries.positive_tab_1, "positive_tab_1")
    gmsh.model.addPhysicalGroup(1, list(lines_vertical[0, [0]]), boundaries.negative_tab_1, "negative_tab_1")
    gmsh.model.addPhysicalGroup(1, list(lines_vertical[-1, [0]]), boundaries.negative_tab_2, "negative_tab_2")

    insulated_negative_cc_1 = list(lines_vertical[0, [-4]]) + list(lines_horizontal[1, [0, 1, 2]])
    insulated_graphite_1 = list(lines_vertical[1, [3, -4]])
    insulated_separator_1 = list(lines_vertical[2, [2, -3]]) + list(lines_horizontal[2, [2, -3]]) + list(lines_horizontal[3, [2, 3, 4, -4, -3]])
    insulated_nmc_1 = list(lines_vertical[3, [4, -5]])
    insulated_positive_cc_1 = list(lines_vertical[4, [3]]) + list(lines_horizontal[4, [3, -4, -3, -2]])  + list(lines_horizontal[5, [3, -4, -3, -2]])
    insulated_nmc_2 = list(lines_vertical[5, [4, -5]])
    insulated_separator_2 = list(lines_vertical[6, [2, -3]]) + list(lines_horizontal[-3, [2, -3]]) + list(lines_horizontal[-4, [2, 3, 4, -4, -3]])
    insulated_graphite_2 = list(lines_vertical[-2, [3, -4]])
    insulated_negative_cc_2 = list(lines_vertical[-1, [-4]]) + list(lines_horizontal[-2, [0, 1, 2]])
    insulated_free_electrolyte = list(lines_vertical[1:-1, 1]) + list(lines_vertical[1:-1, -1]) + list(lines_horizontal[1, -3:]) + list(lines_horizontal[-2, -3:])
    insulated = insulated_negative_cc_1 + insulated_graphite_1 + insulated_separator_1 + insulated_nmc_1 + insulated_positive_cc_1 +\
        insulated_nmc_2 + insulated_separator_2 + insulated_graphite_2 + insulated_negative_cc_2 + insulated_free_electrolyte
    gmsh.model.addPhysicalGroup(1, insulated_negative_cc_1, boundaries.insulated_negative_cc_1, "insulated_negative_cc_1")
    gmsh.model.addPhysicalGroup(1, insulated_graphite_1, boundaries.insulated_graphite_1, "insulated_graphite_1")
    gmsh.model.addPhysicalGroup(1, insulated_separator_1, boundaries.insulated_separator_1, "insulated_separator_1")
    gmsh.model.addPhysicalGroup(1, insulated_nmc_1, boundaries.insulated_nmc_1, "insulated_nmc_1")
    gmsh.model.addPhysicalGroup(1, insulated_positive_cc_1, boundaries.insulated_positive_cc_1, "insulated_positive_cc_1")
    gmsh.model.addPhysicalGroup(1, insulated_nmc_2, boundaries.insulated_nmc_2, "insulated_nmc_2")
    gmsh.model.addPhysicalGroup(1, insulated_separator_2, boundaries.insulated_separator_2, "insulated_separator_2")
    gmsh.model.addPhysicalGroup(1, insulated_graphite_2, boundaries.insulated_graphite_2, "insulated_graphite_2")
    gmsh.model.addPhysicalGroup(1, insulated_negative_cc_2, boundaries.insulated_negative_cc_2, "insulated_negative_cc_2")
    gmsh.model.addPhysicalGroup(1, insulated, boundaries.insulated, "insulated")

    gmsh.model.occ.synchronize()
    negative_cc_1 = list(surfaces[0, :-3])
    graphite_1 = list(surfaces[1, 2:-3])
    separator_1 = list(surfaces[2, 1:-2])
    nmc_1 = list(surfaces[3, 4:-4])
    positive_cc_1 = list(surfaces[4, 2:])
    nmc_2 = list(surfaces[-4, 4:-4])
    separator_2 = list(surfaces[-3, 1:-2])
    graphite_2 = list(surfaces[-2, 2:-3])
    negative_cc_2 = list(surfaces[-1, :-3])
    free_electrolyte = list(surfaces[1, [1, 2, -3, -2, -1]]) + list(surfaces[-2, [1, 2, -3, -2, -1]]) +\
        list(surfaces[2, [1, -2, -1]]) + list(surfaces[-3, [1, -2, -1]]) +\
        list(surfaces[3, [1, 2, 3, -4, -3, -2, -1]]) + list(surfaces[5, [1, 2, 3, -4, -3, -2, -1]]) +\
        list(surfaces[4, [1, 2, -1]])
    valid_surfs = negative_cc_1 + graphite_1 + separator_1 + nmc_1 + positive_cc_1 + nmc_2 + separator_2 + graphite_2 + negative_cc_2 + free_electrolyte
    gmsh.model.addPhysicalGroup(2, negative_cc_1, phases.negative_cc_1, "negative_cc_1")
    gmsh.model.addPhysicalGroup(2, graphite_1, phases.graphite_1, "graphite_1")
    gmsh.model.addPhysicalGroup(2, separator_1, phases.separator_1, "separator_1")
    gmsh.model.addPhysicalGroup(2, nmc_1, phases.nmc_1, "nmc_1")
    gmsh.model.addPhysicalGroup(2, positive_cc_1, phases.positive_cc_1, "positive_cc_1")
    gmsh.model.addPhysicalGroup(2, nmc_2, phases.nmc_2, "nmc_2")
    gmsh.model.addPhysicalGroup(2, separator_2, phases.separator_2, "separator_2")
    gmsh.model.addPhysicalGroup(2, graphite_2, phases.graphite_2, "graphite_2")
    gmsh.model.addPhysicalGroup(2, negative_cc_2, phases.negative_cc_2, "negative_cc_2")
    gmsh.model.addPhysicalGroup(2, free_electrolyte, phases.free_electrolyte, "free_electrolyte")
    gmsh.model.occ.synchronize()

    # line below creates structured mesh, so comment line if want unstructured mesh
    gmsh.model.mesh.setTransfiniteAutomatic([(2, s) for s in valid_surfs], cornerAngle=np.pi/4, recombine=False)

    gmsh.model.mesh.generate(2)
    gmsh.write(output_meshfile)
    gmsh.finalize()
