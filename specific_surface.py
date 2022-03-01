#! /usr/bin/env python3
import os

import argparse
import matplotlib.pyplot as plt
import numpy as np


d = 2
phase_1 = 1
phase_2 = 0


def get_sampling_template(r_max):
    direction = np.random.normal(0, 1, 2)
    origin = (np.random.normal(0, 1, 2) * r_max).astype(int)
    return tuple(origin), direction


def two_point_correlation(im, r=0, r_max=100, phase=1):
    n_tries = 1e3
    n_hits = 0
    for _ in range(int(n_tries)):
        origin, direction = get_sampling_template(r_max)
        destination = tuple((origin + direction * r).astype(int))
        if phase == phase_1:
            if im[origin] and im[destination]:
                n_hits += 1
        elif phase == phase_2:
            if not im[origin] and not im[destination]:
                n_hits += 1
    return n_hits / n_tries


def specific_surface(s1_values, r_values):
    dS2dr = (s1_values[1] - s1_values[0]) / (r_values[1] - r_values[0])
    return - dS2dr * 2 * d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='computes specific area')
    parser.add_argument('--working_dir', help='bmp files directory', required=True)

    args = parser.parse_args()
    working_dir = args.working_dir
    im_files = [os.path.join(working_dir, f) for f in os.listdir(working_dir) if f.endswith(".bmp")]
    n_files = len(im_files)
    r_values = np.arange(0, 50 + 1)
    s1_values = np.zeros(51)
    s2_values = np.zeros(51)
    files_count = 0
    for im_file in im_files:
        img = plt.imread(im_file)
        for idx, r in enumerate(r_values):
            s1_loc = two_point_correlation(img, r, phase_1)
            s2_loc = two_point_correlation(img, r, phase_2)
            s1_values[idx] = s1_values[idx] + s1_loc / n_files
            s2_values[idx] = s2_values[idx] + s2_loc / n_files
    surface_1 = specific_surface(s1_values, r_values)
    surface_2 = specific_surface(s2_values, r_values)
    print("Specific surfaces for phase 1 and 2: ", surface_1, surface_2)

    plt.plot(r_values, s1_values, 'r--')
    plt.plot(r_values, s2_values, 'b--')
    plt.xlabel('r')
    plt.legend([r'$s_1(r)$', r'$s_2(r)$'])
    plt.grid()
    plt.show()
