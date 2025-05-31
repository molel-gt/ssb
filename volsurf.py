import os

import matplotlib.pyplot as plt
import pymeshfix as mf
import numpy as np
import open3d as o3d
import pyvista as pv
import trimesh

SCALING = [0.0858e-6, 0.0858e-6, 0.05e-6]


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
    cam_dir = os.path.join(os.environ["WORK_DIR"], "output/segmentation/cam")
    points = {}
    counter = 0
    L_sep = 175
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
    encoding =  trimesh.voxel.encoding.DenseEncoding(data3d)
    voxels = trimesh.voxel.base.VoxelGrid(encoding)
    print(voxels.volume)
    mesh = voxels.marching_cubes
    trimesh.exchange.export.export_mesh(mesh, f"output/segmentation/cam.stl")
    # print("Generated points")
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

    plotter = pv.Plotter()

    trimesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud)
    trimesh.estimate_normals()
    o3d.io.write_triangle_mesh("cam.stl", trimesh, write_ascii=False, print_progress=True)

    v = np.asarray(trimesh.vertices)
    f = np.array(trimesh.triangles)
    f = np.c_[np.full(len(f), 3), f]

    envelope = pv.PolyData(v, f).clean()
    envelope.actor = plotter.add_mesh(envelope, color='red', show_edges=True, opacity=0.5)

    plotter.show()

    # meshfix = mf.MeshFix(envelope)

    # # Repair also fills holes
    # meshfix.repair(verbose=True)

    # envelope = meshfix.mesh.clean().triangulate()
