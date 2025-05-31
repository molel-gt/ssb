import argparse
import os

import matplotlib.pyplot as plt
import pymeshfix as mf
import numpy as np
import open3d as o3d
import pyvista as pv
import trimesh

import utils

SCALING = [0.0858e-6, 0.0858e-6, 0.2e-6]


def get_valid_coords(coords, limits):
    out_coords = []
    x_lims, y_lims, z_lims = limits
    for (x, y, z) in coords:
        if (x_lims[0] <= x <= x_lims[1]) and (y_lims[0] <= y <= y_lims[1]) and (z_lims[0] <= x <= z_lims[1]):
            out_coords.append((x, y, z))
    return out_coords


def generate_neighboring_subcubes(coord, limits):
    coords = []
    x, y, z = coord
    for zval in [z - 1, z, z + 1]:
        coords.extend([
                      (x, y, zval),
                      (x+1, y, zval),
                      (x+1, y+1, zval),
                      (x, y+1, zval),

                      (x-1, y+1, zval),
                      (x-1, y, zval),
                      (x-1, y-1, zval),
                      (x, y-1, zval),
                      (x+1, y-1, zval),
                      ])

    return get_valid_coords(coords, limits)


def count_neighbors(arr, center):
    """
    Count the number of neigbors at numpy grid location (x-1:x+2,y-1:y+2,z-1:z+2).

    input:
        arr: np.array((nx, ny, nz), dtype=np.bool)
        center: (x, y, z)
    return:
        neighbors: int
    """
    x, y, z = center
    neighors = 0
    for idx in range(-1, 2):
        for idy in range(-1, 2):
            for idz in range(-1, 2):
                try:
                    val = arr[x+idx, y+idy, z+idz]
                except IndexError:
                    val = False
                if val:
                    neighors += 1

    return neighors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='secondary current distribution')
    parser.add_argument('--size', help='Lx-Ly-Lz', required=True, type=str)
    parser.add_argument("--origin", help="where to extract data", nargs='?', const=1, default='0-0-0', type=str)
    parser.add_argument('--scale', help='sx-sy-sz', required=True, type=str)
    parser.add_argument("--phase", help="particulate phase", nargs='?', const=1, default='cam', type=str)
    parser.add_argument("--L_sep", help="separator thickness", nargs='?', const=1, default=100, type=float)
    args = parser.parse_args()
    scaling = [float(v) for v in args.scale.split(",")]
    cam_dir = os.path.join(os.environ["WORK_DIR"], "output/segmentation/cam")
    workdir = os.path.join("output/segmentation", f"{args.phase}/{args.size}/{args.origin}")
    utils.make_dir_if_missing(workdir)
    points = {}
    counter = 0
    L_sep = 15e-6
    nx = 201
    ny = 201
    nz = 201
    img_3d = np.zeros((nx, ny, nz))
    data3d = np.zeros((2 * nx, 2 * ny, 2 * nz), dtype=bool)
    print("Processing segmented images")
    for idx in range(1, nz+1):
        img_file = os.path.join(cam_dir, f"{str(idx).zfill(3)}.tif")
        img = plt.imread(img_file).copy()[:201, :201]
        img[:10, ] = 2
        img_3d[:, :, idx-1] = img[:nx, :ny]
        # xycoords = np.where(np.isclose(img, 2))
        # zcoords = (idx-1) * np.ones(xycoords[0].shape)
        # coords = np.vstack((xycoords[0]*SCALING[0], xycoords[1]*SCALING[1], zcoords*SCALING[2])).T
        # points = np.vstack((points, coords))
    cam_coords = np.array(np.where(np.isclose(img_3d, 2))).T
    print("Generating neighboring cubes")
    for coord in cam_coords:
        x = 2 * coord[0]
        y = 2 * coord[1]
        z = 2 * coord[2]
        cube_coords = generate_neighboring_subcubes((x, y, z), [(0, nx * 2), (0, ny * 2), (0, nz * 2)])
        for new_coord in cube_coords:
            data3d[new_coord] = 1
    print(cam_coords.shape)
    print("Generating surface mesh")
    scale_mat = np.matrix([[scaling[0], 0, 0], [0, scaling[1], 0], [0, 0, scaling[2]] ])
    transform = trimesh.voxel.transforms.Transform(scale_mat)
    encoding =  trimesh.voxel.encoding.DenseEncoding(data3d)
    voxels = trimesh.voxel.base.VoxelGrid(encoding, scale_mat)
    print(voxels.volume)
    print(scale_mat.shape)
    vx = voxels.apply_scale(scale_mat)
    print(voxels.volume, vx.volume)
    mesh = vx.marching_cubes
    trimesh.exchange.export.export_mesh(mesh, os.path.join(workdir, "cam.stl"))
    print("Generated points")
    # cloud = o3d.geometry.PointCloud()
    # point_cloud = np.vstack(np.where(data3d == 1)).T
    # cloud.points = o3d.utility.Vector3dVector(point_cloud)
    # cloud.estimate_normals()

    # # Before Fixing Normals
    # # o3d.visualization.draw_geometries([cloud,], point_show_normal=True)

    # # Fix Normals
    # cloud.orient_normals_consistent_tangent_plane(25)

    # # After Fixing Normals
    # # o3d.visualization.draw_geometries([cloud,], point_show_normal=True)
    # print("Fixed normals")

    # plotter = pv.Plotter()

    # trimesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud)
    # trimesh.estimate_normals()
    # o3d.io.write_triangle_mesh("cam.stl", trimesh, write_ascii=False, print_progress=True)

    # v = np.asarray(trimesh.vertices)
    # f = np.array(trimesh.triangles)
    # f = np.c_[np.full(len(f), 3), f]

    # envelope = pv.PolyData(v, f).clean()
    # envelope.actor = plotter.add_mesh(envelope, color='red', show_edges=True, opacity=0.5)

    # plotter.show()

    # meshfix = mf.MeshFix(envelope)

    # # Repair also fills holes
    # meshfix.repair(verbose=True)

    # envelope = meshfix.mesh.clean().triangulate()
