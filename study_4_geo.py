#!/usr/bin/env python3

import os
import pickle
import subprocess

import argparse
import gmsh
import meshio
import numpy as np

import commons, configs, geometry, utils

markers = commons.SurfaceMarkers()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Volume with Contact Loss")
    parser.add_argument("--grid_extents", help="Nx-Ny-Nz that defines the grid bounds", required=True)
    # parser.add_argument("--contact_map", help="Image to generate contact map", required=True)
    parser.add_argument("--phase", help="0 -> void, 1 -> SE, 2 -> AM", nargs='?', const=1, default=1)
    parser.add_argument("--eps", help="coverage of area at left cc", nargs='?', const=1, default=0.05)
    args = parser.parse_args()
    grid_extents = args.grid_extents
    # contact_img_file = args.contact_map
    phase = args.phase
    eps = float(args.eps)
    scaling = configs.get_configs()['VOXEL_SCALING']
    scale_x = 10e-6  # float(scaling['x'])
    scale_y = 10e-6  # float(scaling['y'])
    scale_z = 10e-6  # float(scaling['z'])
    scale_factor = (scale_x, scale_y, scale_z)
    dp = int(configs.get_configs()['GEOMETRY']['dp'])
    h = float(configs.get_configs()['GEOMETRY']['h'])
    origin_str = 'study_4'
    mesh_dir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], 'study_4', grid_extents, str(eps))
    Nx, Ny, Nz = [float(v) for v in grid_extents.split("-")]
    Lx = Nx - 1
    Ly = Ny - 1
    Lz = Nz - 1
    utils.make_dir_if_missing(mesh_dir)
    contact_points_filepath = os.path.join(mesh_dir, "contact_points.pickle")
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

    # img = io.imread(contact_img_file)
    # contact_points = set()
    # for idx in np.argwhere(np.isclose(img, phase)):
    #     contact_points.add(tuple([int(v) for v in idx] + [0]))

    # with open(contact_points_filepath, "wb") as fp:
    #     pickle.dump(contact_points, fp, protocol=pickle.HIGHEST_PROTOCOL)
    r = 2 * Lx * (eps/np.pi) ** 0.5
    xc, yc = 0.5 * Lx, 0.5 * Ly
    res0 = subprocess.check_call(f'cp contact-loss.geo {mesh_dir}', shell=True)
    res1 = subprocess.check_call(f'sed -i "/eps\ = */c\eps = {eps};" {mesh_dir}/contact-loss.geo', shell=True)
    res2 = subprocess.check_call(f'sed -i "/Lx\ = */c\Lx = {Lx};" {mesh_dir}/contact-loss.geo', shell=True)
    res3 = subprocess.check_call(f'sed -i "/Ly\ = */c\Ly = {Ly};" {mesh_dir}/contact-loss.geo', shell=True)
    res4 = subprocess.check_call(f'sed -i "/Lz\ = */c\Lz = {Lz};" {mesh_dir}/contact-loss.geo', shell=True)
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
    print(f"Wrote meshfile '{tria_xdmf_scaled}'")
