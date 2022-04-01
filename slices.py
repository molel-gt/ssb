#!/usr/bin/env python3
import os

import argparse

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from examples import spheres


def build_slice(at_x, centers, radius, yz_size):
    """
    :param at_x:
    :param centers:
    :param radius:
    :param yz_size:
    """
    slice = np.zeros(yz_size, dtype=np.uint8)
    for center in centers:
        for idx, _ in np.ndenumerate(slice):
            if ((at_x - center[0]) ** 2 + (idx[0] - center[1]) ** 2 + (idx[1] - center[2]) ** 2) <= radius ** 2:
                slice[idx] = 255

    return slice


def slice_to_file(slice, fname):
    """
    :param slice:
    :param fname:
    """
    im = Image.fromarray(slice)
    im.save(fname, format='bmp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates 2D slices of packed spheres')
    parser.add_argument('--grid_info', help='Nx-Ny-Nz', required=True, type=str)
    parser.add_argument('--centers_file', help='file with spheres centers', type=str, required=True)
    args = parser.parse_args()
    
    Nx, Ny, Nz = map(lambda x: int(x), args.grid_info.split("-"))
    centers, radius, n = spheres.read_spheres_position_file(args.centers_file)
    centers = list(map(lambda x: (int(x[0] * Nx), int(x[1] * Ny), int(x[2] * Nz)), centers))
    radius = int(radius * Nx)

    print(n, "spheres of radius", radius)
    print("sphere surface area:", 4 * np.pi * radius ** 2)

    for at_x in range(Nx):
        slice = build_slice(at_x, centers, radius, (Ny, Nz))
        fname = os.path.join('spheres', 'S' + str(at_x).zfill(3) + '.bmp')
        slice_to_file(slice, fname)
