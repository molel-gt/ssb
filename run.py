#!/usr/bin/env python3

import os
import shlex
import subprocess
import sys

import utils


if __name__ == '__main__':
    grid_sizes = [int(v) for v in sys.argv[1].split(',')]
    n_files = int(sys.argv[2])
    size_delta = int(sys.argv[3])
    working_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
    extents = utils.get_samples_of_test_grids(grid_sizes, n_files, size_delta)
    for (grid_size, start_pos, stop_pos) in extents:
        grid_info = '_'.join(map(str, [int(grid_size), int(start_pos), int(stop_pos)]))
        proc1 = subprocess.Popen(["./geometry.sh", working_dir, 'Spheres_3',  '90_90', grid_info,])
        proc1.wait()
        proc2 = subprocess.Popen(
                       shlex.split('mpirun -np 2 '
                       'python3 ion_transport.py --working_dir={}'
                       ' --grid_info={} --file_shape=90_90'.format(
                           working_dir, grid_info))
                           )
        proc2.wait()
