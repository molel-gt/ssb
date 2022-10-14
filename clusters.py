#! /usr/bin/env python3

import gc
import os
import pickle
import subprocess
import time

import argparse
import gmsh
import logging
import meshio
import networkx as nx
import numpy as np

from skimage import io

import configs, commons, constants, filter_voxels, geometry, utils


FORMAT = '%(asctime)s: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__file__)
logger.setLevel('INFO')

phase_key = constants.phase_key
surface_tags = constants.surface_tags
CELL_TYPES = commons.CellTypes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='computes specific area')
    parser.add_argument('--grid_info', help='Nx-Ny-Nz',
                        required=True)
    parser.add_argument("--phase", help='Phase that we want to reconstruct, e.g. 0 for void, 1 for solid electrolyte and 2 for active material', nargs='?', const=1, default=1, type=int)

    args = parser.parse_args()
    scaling = configs.get_configs()['VOXEL_SCALING']
    img_folder = configs.get_configs()['LOCAL_PATHS']['segmented_image_stack']
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])
    scale_factor = (scale_x, scale_y, scale_z)
    origin = [int(v) for v in configs.get_configs()['GEOMETRY']['origin'].split(",")]
    origin_str = "-".join([str(v).zfill(3) for v in origin])
    grid_info = args.grid_info
    grid_size = int(args.grid_info.split("-")[0])
    Nx, Ny, Nz = [int(v) for v in args.grid_info.split("-")]
    Lx = Nx - 1
    Ly = Ny - 1
    Lz = Nz - 1
    mesh_dir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], f"{args.phase}/{grid_info}_{origin_str}/clusters")
    utils.make_dir_if_missing(mesh_dir)

    im_files = sorted([os.path.join(img_folder, f) for
                       f in os.listdir(img_folder) if f.endswith(".tif")])
    n_files = len(im_files)

    start_time = time.time()

    shape = [*io.imread(im_files[0]).shape, n_files]
    voxels_raw = filter_voxels.load_images(im_files, shape)[origin[0]:Nx+origin[0], origin[1]:Ny+origin[1], origin[2]:Nz+origin[2]]
    voxels_filtered = filter_voxels.get_filtered_voxels(voxels_raw)
    voxels = np.isclose(voxels_filtered, args.phase)

    points0 = geometry.build_points(voxels, dp=1)
    # points = geometry.add_boundary_points(points, x_max=Lx, y_max=Ly, z_max=Lz, h=0.5, dp=1)
    points_view0 = {v: k for k, v in points0.items()}

    G = geometry.build_graph(points0)
    pieces = nx.connected_components(G)
    pieces = [piece for piece in pieces]
    logger.info("{:,} components".format(len(pieces)))
    for idx, piece in enumerate(pieces):
        working_piece = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
        piece_id = str(idx).zfill(6)
        for p in piece:
            coord = tuple(points_view0[p])
            working_piece[coord] = 1
        if np.all(working_piece[:, 0, :] == 0) or np.all(working_piece[:, Ly, :] == 0):
            logger.debug(f"Piece {idx} does not span both ends")
            continue
        logger.info(f"Piece {piece_id} spans both ends along y-axis")
        utils.make_dir_if_missing(os.path.join(mesh_dir, piece_id))
        points = geometry.build_points(working_piece, dp=1)
        points = geometry.add_boundary_points(points, x_max=Lx, y_max=Ly, z_max=Lz, h=0.5, dp=1)
        points = geometry.add_boundary_points(points, x_max=Lx, y_max=Ly, z_max=Lz, h=0.5, dp=1)
        points_view = {v: k for k, v in points.items()}

        neighbors = geometry.number_of_neighbors(voxels)
        effective_electrolyte = geometry.electrolyte_bordering_active_material(voxels_filtered, dp=1)
        effective_electrolyte = geometry.extend_points(effective_electrolyte, points, x_max=Nx-1, y_max=Ny-1, z_max=Nz-1, h=0.5, dp=1)
        eff_electrolyte_filepath = os.path.join(mesh_dir, piece_id, "effective_electrolyte.pickle")
        with open(eff_electrolyte_filepath, "wb") as fp:
            pickle.dump(effective_electrolyte, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        cubes = geometry.build_variable_size_cubes(points, h=0.5)
        tetrahedra = geometry.build_tetrahedra(cubes, points, points_view)

        nodefile = os.path.join(mesh_dir, piece_id, "porous.node")
        tetfile = os.path.join(mesh_dir, piece_id, "porous.ele")
        facesfile = os.path.join(mesh_dir, piece_id, "porous.face")
        vtkfile = os.path.join(mesh_dir, piece_id, "porous.1.vtk")
        surface_vtk = os.path.join(mesh_dir, piece_id, "surface.vtk")
        tetr_mshfile = os.path.join(mesh_dir, piece_id, "porous_tetr.msh")
        surf_mshfile = os.path.join(mesh_dir, piece_id, "porous_tria.msh")
        tetr_xdmf_scaled = os.path.join(mesh_dir,piece_id,  "tetr.xdmf")
        tetr_xdmf_unscaled = os.path.join(mesh_dir, piece_id, "tetr_unscaled.xdmf")
        tria_xdmf_scaled = os.path.join(mesh_dir, piece_id, "tria.xdmf")
        tria_xdmf_unscaled = os.path.join(mesh_dir, piece_id, "tria_unscaled.xdmf")
        tria_xmf_unscaled = os.path.join(mesh_dir, piece_id, "tria_unscaled.xmf")

        geometry.write_points_to_node_file(nodefile, points)
        geometry.write_tetrahedra_to_ele_file(tetfile, tetrahedra)

        # Free up memory of objects we won't use
        tetrahedra = None
        cubes = None
        voxels = None
        gc.collect()

        # TetGen
        retcode_tetgen = subprocess.check_call(f"tetgen {tetfile} -rkQF", shell=True)

        # GMSH
        counter = 0
        gmsh.initialize()
        gmsh.model.add("porous")
        gmsh.merge(vtkfile)
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(dim=3)
        for i, volume in enumerate(volumes):
            marker = int(counter + i)
            gmsh.model.addPhysicalGroup(3, [volume[1]], marker)
            gmsh.model.setPhysicalName(3, marker, f"V{marker}")
        gmsh.model.occ.synchronize()

        gmsh.model.mesh.generate(3)
        gmsh.write(tetr_mshfile)
        gmsh.finalize()

        tet_msh = meshio.read(tetr_mshfile)
        tetr_mesh_unscaled = geometry.create_mesh(tet_msh, CELL_TYPES.tetra)
        tetr_mesh_unscaled.write(tetr_xdmf_unscaled)
        tetr_mesh_scaled = geometry.scale_mesh(tetr_mesh_unscaled, CELL_TYPES.tetra, scale_factor=scale_factor)
        tetr_mesh_scaled.write(tetr_xdmf_scaled)

        retcode_paraview = subprocess.check_call("pvpython extract_surface_from_volume.py {}".format(os.path.dirname(tetr_xdmf_unscaled)), shell=True)
        surf_msh = meshio.read(tria_xmf_unscaled)
        tria_mesh_unscaled = geometry.label_surface_mesh(surf_msh, effective_electrolyte, Ny - 1)
        tria_mesh_unscaled.write(tria_xdmf_unscaled)

        tria_mesh_scaled = geometry.scale_mesh(tria_mesh_unscaled, CELL_TYPES.triangle, scale_factor=scale_factor)
        tria_mesh_scaled.write(tria_xdmf_scaled)
        logger.info(f"Wrote tetr.xdmf and tria.xdmf mesh files for piece {piece_id}")