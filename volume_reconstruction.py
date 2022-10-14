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
import numpy as np

from skimage import io

import constants, commons, configs, filter_voxels, geometry, utils


FORMAT = '%(asctime)s: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__file__)
logger.setLevel(configs.get_configs()['LOGGING']['level'])

phase_key = constants.phase_key
surface_tags = constants.surface_tags
CELL_TYPES = commons.CellTypes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reconstructs volume from segemented images.')
    parser.add_argument('--grid_info', help='Grid size is given by Nx-Ny-Nz such that the lengths are (Nx-1) by (Ny - 1) by (Nz - 1).',
                        required=True)
    parser.add_argument("--phase", help='Phase that we want to reconstruct, e.g. 0 for void, 1 for solid electrolyte and 2 for active material', nargs='?', const=1, default=1, type=int)
    start_time = time.time()
    args = parser.parse_args()
    phase = args.phase
    scaling = configs.get_configs()['VOXEL_SCALING']
    img_folder = configs.get_configs()['LOCAL_PATHS']['segmented_image_stack']
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])
    scale_factor = (scale_x, scale_y, scale_z)
    origin = [int(v) for v in configs.get_configs()['GEOMETRY']['origin'].split(",")]
    origin_str = "-".join([str(v).zfill(3) for v in origin])
    grid_info = "-".join([v.zfill(3) for v in args.grid_info.split("-")])
    grid_size = int(args.grid_info.split("-")[0])
    Nx, Ny, Nz = [int(v) for v in args.grid_info.split("-")]
    Lx = Nx - 1
    Ly = Ny - 1
    Lz = Nz - 1
    mesh_dir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], f"{phase}/{grid_info}_{origin_str}")
    utils.make_dir_if_missing(mesh_dir)
    im_files = sorted([os.path.join(img_folder, f) for
                       f in os.listdir(img_folder) if f.endswith(".tif")])
    n_files = len(im_files)

    start_time = time.time()

    shape = [*io.imread(im_files[0]).shape, n_files]
    voxels_raw = filter_voxels.load_images(im_files, shape)[origin[0]:Nx+origin[0], origin[1]:Ny+origin[1], origin[2]:Nz+origin[2]]
    voxels_filtered = filter_voxels.get_filtered_voxels(voxels_raw)
    voxels = np.isclose(voxels_filtered, phase)

    points = geometry.build_points(voxels, dp=1)
    points = geometry.add_boundary_points(points, x_max=Nx-1, y_max=Ny-1, z_max=Nz-1, h=0.5, dp=1)
    points_view = {v: k for k, v in points.items()}

    neighbors = geometry.number_of_neighbors(voxels)
    effective_electrolyte = geometry.electrolyte_bordering_active_material(voxels_filtered, dp=1)
    effective_electrolyte = geometry.extend_points(effective_electrolyte, points, x_max=Nx-1, y_max=Ny-1, z_max=Nz-1, h=0.5, dp=1)
    eff_electrolyte_filepath = os.path.join(mesh_dir, "effective_electrolyte.pickle")
    with open(eff_electrolyte_filepath, "wb") as fp:
        pickle.dump(effective_electrolyte, fp, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Rough porosity : {:0.4f}".format(np.sum(voxels) / (Nx * Ny * Nz)))
    # Only label points that will be used for meshing
    n_tetrahedra = 0
    n_triangles = 0
    counter = 0
    tetrahedra = {}
    
    cubes = geometry.build_variable_size_cubes(points, h=0.5)
    for cube in cubes:
        _tetrahedra = geometry.build_tetrahedra(cube, points)
        for tet in _tetrahedra:
            tetrahedra[tet] = n_tetrahedra
            n_tetrahedra += 1
    tetrahedra_np = np.zeros((n_tetrahedra, 12))
    points_set = set()
    for i, tet in enumerate(list(tetrahedra.keys())):
        for j, vertex in enumerate(tet):
            coord = points_view[vertex]
            tetrahedra_np[i, 3*j:3*j+3] = coord
            points_set.add(tuple(coord))

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

    with open(nodefile, "w") as fp:
        fp.write("%d 3 0 0\n" % int(len(points.values())))
        for coord, point_id in points.items():
            x0, y0, z0 = coord
            fp.write(f"{point_id} {x0} {y0} {z0}\n")
    tet_points = set()
    with open(tetfile, "w") as fp:
        fp.write(f"{n_tetrahedra} 4 0\n")
        for tet_id, tetrahedron in enumerate(tetrahedra_np):
            p1 = points[tuple(tetrahedron[:3])]
            p2 = points[tuple(tetrahedron[3:6])]
            p3 = points[tuple(tetrahedron[6:9])]
            p4 = points[tuple(tetrahedron[9:])]
            tet_points |= {p1, p2, p3, p4}
            fp.write(f"{tet_id} {p1} {p2} {p3} {p4}\n")

    # Free up memory of objects we won't use
    tetrahedra = None
    tetrahedra_np = None
    cubes = None
    voxels = None
    gc.collect()

    retcode_tetgen = subprocess.check_call(f"tetgen {tetfile} -rkQF", shell=True)

    # GMSH
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
    
    logger.info("Took {:,} seconds".format(int(time.time() - start_time)))
