#! /usr/bin/env python3

import gc
import os
import pickle
import subprocess
import time

import argparse
import gmsh
import logging
import matplotlib.pyplot as plt
import meshio
import numpy as np

from dolfinx import cpp, io, mesh
from mpi4py import MPI

import commons, configs, filter_voxels, geometry, utils


FORMAT = '%(asctime)s: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__file__)
logger.setLevel(configs.get_configs()['LOGGING']['level'])

CELL_TYPES = commons.CellTypes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reconstructs volume from segemented images.')
    parser.add_argument('--grid_extents', help='Grid size is given by Nx-Ny-Nz such that the lengths are (Nx-1) by (Ny - 1) by (Nz - 1).',
                        required=True)
    parser.add_argument("--phase", help='0 - VOID, 1 - SE, 2 - AM', nargs='?', const=1, default=1, type=int)
    parser.add_argument("--segmentation_folder", help='folder containing segmented stack', nargs='?', const=1,
                        default='segmented', type=str)
    parser.add_argument("--output_folder", help='parent folder for output files', nargs='?', const=1,
                        default='mesh/study_1', type=str)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='VOXEL_SCALING', type=str)
    start_time = time.time()
    args = parser.parse_args()
    phase = args.phase
    scaling = configs.get_configs()[args.scaling]
    img_folder = args.segmentation_folder
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])
    dp = int(configs.get_configs()['GEOMETRY']['dp'])
    h = float(configs.get_configs()['GEOMETRY']['h'])
    scale_factor = (scale_x, scale_y, scale_z)
    origin_str = args.grid_extents.split("_")[1]
    origin = [int(v) for v in origin_str.split("-")]
    grid_extents = args.grid_extents.split("_")[0]
    grid_size = int(args.grid_extents.split("-")[0])
    Nx, Ny, Nz = [int(v) for v in grid_extents.split("-")]
    Lx = Nx - 1
    Ly = Ny - 1
    Lz = Nz - 1
    markers = commons.SurfaceMarkers()
    mesh_dir = os.path.join(args.output_folder, args.segmentation_folder, f"{args.grid_extents}/{phase}")
    utils.make_dir_if_missing(mesh_dir)
    im_files = sorted([os.path.join(img_folder, f) for
                       f in os.listdir(img_folder) if f.endswith(".tif")])
    n_files = len(im_files)

    start_time = time.time()

    shape = [*plt.imread(im_files[0]).shape, n_files]
    voxels_raw = filter_voxels.load_images(im_files, shape)[origin[0]:Nx+origin[0], origin[1]:Ny+origin[1], origin[2]:Nz+origin[2]]
    voxels_filtered = filter_voxels.get_filtered_voxels(voxels_raw)
    voxels = np.isclose(voxels_filtered, phase)

    logger.info("Rough porosity : {:0.4f}".format(np.sum(voxels) / (Nx * Ny * Nz)))

    points = geometry.build_points(voxels, dp=dp)
    points = geometry.add_boundary_points(points, x_max=Lx, y_max=Ly, z_max=Lz, h=h, dp=dp)
    cubes = geometry.build_variable_size_cubes(points, h=h, dp=dp)
    tetrahedra = geometry.build_tetrahedra(cubes, points)
    points_view = {v: k for k, v in points.items()}

    effective_electrolyte = geometry.electrolyte_bordering_active_material(voxels_filtered, dp=dp)
    effective_electrolyte = geometry.extend_points(effective_electrolyte, points, x_max=Lx, y_max=Ly, z_max=Lz, h=h, dp=dp)
    eff_electrolyte_filepath = os.path.join(mesh_dir, "effective_electrolyte.pickle")
    with open(eff_electrolyte_filepath, "wb") as fp:
        pickle.dump(effective_electrolyte, fp, protocol=pickle.HIGHEST_PROTOCOL)

    nodefile = os.path.join(mesh_dir, "porous.node")
    edgefile = os.path.join(mesh_dir, "porous.edge")
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

    geometry.write_points_to_node_file(nodefile, points)
    geometry.write_tetrahedra_to_ele_file(tetfile, tetrahedra)

    # Free up memory of objects we won't use
    tetrahedra = None
    cubes = None
    voxels = None
    gc.collect()

    # TetGen
    retcode_tetgen = subprocess.check_call(f"tetgen {tetfile} -rkQR", shell=True)

    # GMSH
    counter = 0
    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', 1)
    gmsh.model.add("porous")
    gmsh.merge(vtkfile)
    gmsh.model.occ.synchronize()

    volumes = [v[1] for v in gmsh.model.getEntities(dim=3)]
    gmsh.model.addPhysicalGroup(3, volumes, phase)
    gmsh.model.occ.synchronize()

    left = []
    right = []
    insulated = []
    for surface in gmsh.model.getEntities(2):
        com = gmsh.model.occ.getCenterOfMass(*surface)
        if np.isclose(com[2], 0):
            left.append(surface[1])
        elif np.isclose(com[2], Lz):
            right.append(surface[1])
        else:
            insulated.append(surface[1])
    gmsh.model.addPhysicalGroup(2, left, markers.left_cc)
    gmsh.model.addPhysicalGroup(2, right, markers.right_cc)
    gmsh.model.addPhysicalGroup(2, insulated, markers.insulated)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(tetr_mshfile)
    gmsh.finalize()

    msh = meshio.read(tetr_mshfile)
    tetr_mesh_unscaled = geometry.create_mesh(msh, CELL_TYPES.tetra)
    tetr_mesh_unscaled.write(tetr_xdmf_unscaled)
    tetr_mesh_scaled = geometry.scale_mesh(tetr_mesh_unscaled, CELL_TYPES.tetra, scale_factor=scale_factor)
    tetr_mesh_scaled.write(tetr_xdmf_scaled)

    # tria_mesh_unscaled = geometry.create_mesh(msh, CELL_TYPES.triangle)
    # tria_mesh_unscaled.write(tria_xdmf_unscaled)
    # tria_mesh_scaled = geometry.scale_mesh(tria_mesh_unscaled, CELL_TYPES.triangle, scale_factor=scale_factor)
    # tria_mesh_scaled.write(tria_xdmf_scaled)

    for f in [nodefile, edgefile, tetfile, facesfile, vtkfile, surface_vtk, tetr_mshfile, surf_mshfile, tetr_xdmf_unscaled, tria_xdmf_unscaled]:
        try:
            os.remove(f)
            new_f = f.split('.')[0] + '.1.' + f.split('.')[-1]
            os.remove(new_f)
        except:
            continue
    
    logger.info("Took {:,} seconds".format(int(time.time() - start_time)))
