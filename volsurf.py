import os

import matplotlib.pyplot as plt
import pymeshfix as mf
import numpy as np
import open3d as o3d
import pyvista as pv

SCALING = [0.0858e-6, 0.0858e-6, 0.05e-6]


if __name__ == '__main__':
    cam_dir = os.path.join(os.environ["WORK_DIR"], "output/segmentation/cam")
    points = np.empty((0, 3))
    for idx in range(1, 201):
        img_file = os.path.join(cam_dir, f"{str(idx).zfill(3)}.tif")
        img = plt.imread(img_file).copy()[:201, :201]
        img[:10, ] = 2
        xycoords = np.where(np.isclose(img, 2))
        zcoords = (idx-1) * np.ones(xycoords[0].shape)
        coords = np.vstack((xycoords[0]*SCALING[0], xycoords[1]*SCALING[1], zcoords*SCALING[2])).T
        points = np.vstack((points, coords))
    print("Generated points")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals()

    # Before Fixing Normals
    # o3d.visualization.draw_geometries([cloud,], point_show_normal=True)

    # Fix Normals
    cloud.orient_normals_consistent_tangent_plane(25)

    # After Fixing Normals
    # o3d.visualization.draw_geometries([cloud,], point_show_normal=True)
    print("Fixed normals")

    plotter = pv.Plotter()

    trimesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud)

    v = np.asarray(trimesh.vertices)
    f = np.array(trimesh.triangles)
    f = np.c_[np.full(len(f), 3), f]

    envelope = pv.PolyData(v, f).clean()
    envelope.actor = plotter.add_mesh(envelope, color='red', show_edges=True, opacity=0.5)

    plotter.show()

    meshfix = mf.MeshFix(envelope)

    # Repair also fills holes
    meshfix.repair(verbose=True)

    envelope = meshfix.mesh.clean().triangulate()
