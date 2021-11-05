#!/usr/bin/env python3

import numpy as np


def get_samples_of_test_grids(grid_sizes, n_files, size_delta):
    """
    Obtains indices to use in order to produce a cubes of lengths `grid_sizes`
    from `n_files` stack of images.
    Consecutive start indices are size `size_delta` apart
    """
    test_grids = np.empty([0, 3])
    for grid_size in grid_sizes:
        if n_files < grid_size:
            raise Exception("cannot extract the grid because n_files < grid_size")
        num_grids = int((n_files - grid_size)/size_delta)
        grid_extents = np.empty([num_grids, 3])
        for i_x in range(num_grids):
            start_pos = int(size_delta * i_x)
            end_pos = int(size_delta * i_x + grid_size)
            if n_files - start_pos < grid_size:
                continue
            grid_extents[i_x, :] = [grid_size, start_pos, end_pos]
        test_grids = np.vstack((test_grids, grid_extents))

    return test_grids
