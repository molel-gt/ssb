#!/usr/bin/env python3

import os
import json
import subprocess

import argparse
import gmsh
import meshio
import numpy as np

import commons, configs, geometry, utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Volume with Contact Loss")
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="contact_loss_ref")
    parser.add_argument("--dimensions", help="integer representation of Lx-Ly-Lz of the grid", required=True)
    parser.add_argument("--eps", help="coverage of area at left cc", nargs='?', const=1, default=0.05, type=float)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='MICRON_TO_METER', type=str)
    parser.add_argument("--resolution", help="maximum resolution", nargs='?', const=1, default=1, type=float)
    args = parser.parse_args()
    markers = commons.Markers()
    dimensions = args.dimensions
    scaling = configs.get_configs()[args.scaling]
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])
    resolution = args.resolution * scale_x

    mesh_dir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], args.name_of_study, args.dimensions, str(args.eps), str(args.resolution))
    Lx, Ly, Lz = [int(v) for v in args.dimensions.split("-")]
    Lx = Lx * scale_x
    Ly = Ly * scale_y
    Lz = Lz * scale_z
    utils.make_dir_if_missing(mesh_dir)
    geometry_metafile = os.path.join(mesh_dir, "geometry.json")
    tetr_mshfile = os.path.join(mesh_dir, "trial.msh")

    _ = subprocess.check_call(f'cp contact-loss-ref.geo {mesh_dir}', shell=True)
    _ = subprocess.check_call(f'sed -i "/eps\ = */c\eps = {args.eps};" {mesh_dir}/contact-loss-ref.geo', shell=True)
    _ = subprocess.check_call(f'sed -i "/Lx\ = */c\Lx = {Lx};" {mesh_dir}/contact-loss-ref.geo', shell=True)
    _ = subprocess.check_call(f'sed -i "/Ly\ = */c\Ly = {Ly};" {mesh_dir}/contact-loss-ref.geo', shell=True)
    _ = subprocess.check_call(f'sed -i "/Lz\ = */c\Lz = {Lz};" {mesh_dir}/contact-loss-ref.geo', shell=True)
    _ = subprocess.check_call(f'sed -i "/Lz\ = */c\Lz = {Lz};" {mesh_dir}/contact-loss-ref.geo', shell=True)
    _ = subprocess.check_call(f'sed -i "/lmax\ = */c\lmax = {resolution};" {mesh_dir}/contact-loss-ref.geo', shell=True)
    _ = subprocess.check_call(f'sed -i "/left\ = */c\left = {markers.left};" {mesh_dir}/contact-loss-ref.geo', shell=True)
    _ = subprocess.check_call(f'sed -i "/right\ = */c\right = {markers.right};" {mesh_dir}/contact-loss-ref.geo', shell=True)
    _ = subprocess.check_call(f'sed -i "/insulated\ = */c\insulated = {markers.insulated};" {mesh_dir}/contact-loss-ref.geo', shell=True)
    res = subprocess.check_call(f"gmsh -3 {mesh_dir}/contact-loss-ref.geo -o {tetr_mshfile}", shell=True)
    geometry_metadata = {
        "max_resolution": args.resolution,
        "dimensions": args.dimensions,
        "scaling": args.scaling,
    }
    with open(geometry_metafile, "w", encoding='utf-8') as f:
        json.dump(geometry_metadata, f, ensure_ascii=False, indent=4)
    print(f"Wrote meshfile '{tetr_mshfile}'")
