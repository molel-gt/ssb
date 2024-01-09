#!/usr/bin/env python3

import os
import json
import subprocess

import argparse
import gmsh
import meshio
import numpy as np

import commons, configs, geometry, utils, study_4_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Volume with Contact Loss")
    parser.add_argument("--dimensions", help="integer representation of Lx-Ly-Lz of the grid", required=True)
    parser.add_argument("--eps", help="coverage of area at left cc", nargs='?', const=1, default=0.05)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='STUDY4_VOXEL_SCALING', type=str)
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="study_4")
    parser.add_argument("--max_resolution", help="maximum resolution", nargs='?', const=1, default=1)
    args = parser.parse_args()
    markers = commons.SurfaceMarkers()
    dimensions = args.dimensions
    max_resolution = args.max_resolution
    # contact_img_file = args.contact_map
    eps = float(args.eps)
    scaling = configs.get_configs()[args.scaling]
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])
    scale_factor = (scale_x, scale_y, scale_z)

    dp = int(configs.get_configs()['GEOMETRY']['dp'])
    h = float(configs.get_configs()['GEOMETRY']['h'])
    origin_str = args.name_of_study
    mesh_dir = study_4_utils.meshfile_subfolder_path(eps=args.eps, name_of_study=args.name_of_study, dimensions=args.dimensions, max_resolution=args.max_resolution)
    Lx, Ly, Lz = [int(v) for v in dimensions.split("-")]
    utils.make_dir_if_missing(mesh_dir)
    geometry_metafile = os.path.join(mesh_dir, "geometry.json")
    nodefile = os.path.join(mesh_dir, "porous.node")
    tetfile = os.path.join(mesh_dir, "porous.ele")
    facesfile = os.path.join(mesh_dir, "porous.face")
    vtkfile = os.path.join(mesh_dir, "porous.1.vtk")
    surface_vtk = os.path.join(mesh_dir, "surface.vtk")
    tetr_mshfile = os.path.join(mesh_dir, "porous_tetr.msh")
    surf_mshfile = os.path.join(mesh_dir, "porous_tria.msh")
    tetr_xdmf_scaled = os.path.join(mesh_dir, "tetr.xdmf")
    tetr_xdmf_unscaled = os.path.join(mesh_dir, "tetr_unscaled.xdmf")
    tria_xdmf_scaled = os.path.join(mesh_dir, "tria.xdmf")
    tria_xdmf_unscaled = os.path.join(mesh_dir, "tria_unscaled.xdmf")
    tria_xmf_unscaled = os.path.join(mesh_dir, "tria_unscaled.xmf")

    r = 2 * Lx * (eps/np.pi) ** 0.5
    xc, yc = 0.5 * Lx, 0.5 * Ly
    _ = subprocess.check_call(f'cp contact-loss.geo {mesh_dir}', shell=True)
    _ = subprocess.check_call(f'sed -i "/eps\ = */c\eps = {eps};" {mesh_dir}/contact-loss.geo', shell=True)
    _ = subprocess.check_call(f'sed -i "/Lx\ = */c\Lx = {Lx};" {mesh_dir}/contact-loss.geo', shell=True)
    _ = subprocess.check_call(f'sed -i "/Ly\ = */c\Ly = {Ly};" {mesh_dir}/contact-loss.geo', shell=True)
    _ = subprocess.check_call(f'sed -i "/Lz\ = */c\Lz = {Lz};" {mesh_dir}/contact-loss.geo', shell=True)
    _ = subprocess.check_call(f'sed -i "/Lz\ = */c\Lz = {Lz};" {mesh_dir}/contact-loss.geo', shell=True)
    _ = subprocess.check_call(f'sed -i "/lmax\ = */c\lmax = {max_resolution};" {mesh_dir}/contact-loss.geo', shell=True)
    res = subprocess.check_call(f"gmsh -3 {mesh_dir}/contact-loss.geo -o {tetr_mshfile}", shell=True)

    tet_msh = meshio.read(tetr_mshfile)
    tetr_mesh_unscaled = geometry.create_mesh(tet_msh, "tetra")
    tetr_mesh_unscaled.write(tetr_xdmf_unscaled)
    tetr_mesh_scaled = geometry.scale_mesh(tetr_mesh_unscaled, "tetra", scale_factor=scale_factor)
    tetr_mesh_scaled.write(tetr_xdmf_scaled)
    print(f"Wrote meshfile '{tetr_xdmf_scaled}'")
    tria_mesh_unscaled = geometry.create_mesh(tet_msh, "triangle")
    tria_mesh_unscaled.write(tria_xdmf_unscaled)
    tria_mesh_scaled = geometry.scale_mesh(tria_mesh_unscaled, "triangle", scale_factor=scale_factor)
    tria_mesh_scaled.write(tria_xdmf_scaled)
    geometry_metadata = {
        "max_resolution": args.max_resolution,
        "dimensions": args.dimensions,
        "scaling": args.scaling,
    }
    with open(geometry_metafile, "w", encoding='utf-8') as f:
        json.dump(geometry_metadata, f, ensure_ascii=False, indent=4)
    print(f"Wrote meshfile '{tria_xdmf_scaled}'")
