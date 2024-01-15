#!/usr/bin/env python3
import argparse
import json
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coupling of stress and lithium metal/electrolyte active area fraction.')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="separator_mechanics")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--voltage", help="applied voltage drop", nargs='?', const=1, default=1e-3)
    parser.add_argument("--Wa", help="Wagna number: charge transfer resistance <over> ohmic resistance", nargs='?', const=1, default=np.inf)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='VOXEL_SCALING', type=str)

    args = parser.parse_args()
    data_dir = os.path.join(f'{args.mesh_folder}')