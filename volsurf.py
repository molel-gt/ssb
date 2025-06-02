import argparse
import os

import matplotlib.pyplot as plt
import pymeshfix as mf
import numpy as np
import trimesh

import utils


def get_valid_coords(coords, limits):
    out_coords = []
    x_lims, y_lims, z_lims = limits
    for (x, y, z) in coords:
        if (x_lims[0] <= x <= x_lims[1]) and (y_lims[0] <= y <= y_lims[1]) and (z_lims[0] <= z <= z_lims[1]):
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


def generate_surface_mesh_for_phase(img_3d, sizes):
    """
    :rtype:
        trimesh.Trimesh
    """
    nx, ny, nz = sizes
    # data3d = np.zeros((2 * nx, 2 * ny, 2 * nz), dtype=bool)
    phase_coords = np.array(np.where(img_3d)).T
    print("Generating neighboring cubes")
    coords = []
    for coord in phase_coords:
        x = 2 * coord[0]
        y = 2 * coord[1]
        z = 2 * coord[2]
        cube_coords = generate_neighboring_subcubes((x, y, z), [(0, nx * 2), (0, ny * 2), (0, nz * 2)])
        for new_coord in cube_coords:
            coords.append(new_coord)
            # data3d[new_coord] = 1
    print("Generating surface mesh")
    pc = trimesh.PointCloud(np.array(coords))
    pitch = 2
    mesh = trimesh.voxel.ops.points_to_marching_cubes(pc.vertices, pitch=pitch)
    # encoding =  trimesh.voxel.encoding.DenseEncoding(data3d)
    # voxels = trimesh.voxel.base.VoxelGrid(encoding)
    # mesh = voxels.marching_cubes

    return mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='secondary current distribution')
    parser.add_argument('--size', help='grid size Lx-Ly-Lz', required=True, type=str)
    parser.add_argument("--origin", help="where to extract data", nargs='?', const=1, default='0-0-0', type=str)
    parser.add_argument('--scale', help='sx-sy-sz', required=True, type=str)
    parser.add_argument("--L_sep", help="separator thickness", nargs='?', const=1, default=15e-6, type=float)
    args = parser.parse_args()
    trimesh.util.attach_to_log()
    scaling = [float(v) for v in args.scale.split(",")]
    cam_dir = os.path.join(os.environ["WORK_DIR"], "output/segmentation/cam")
    voids_dir = os.path.join(os.environ["WORK_DIR"], "output/segmentation/voids")
    workdir = os.path.join("output/segmentation", f"{args.size}/{args.origin}")
    utils.make_dir_if_missing(workdir)
    cam_dir = os.path.join(os.environ["WORK_DIR"], "output/segmentation/cam")
    voids_dir = os.path.join(os.environ["WORK_DIR"], "output/segmentation/voids")

    nx, ny, nz = [int(v) for v in args.size.split("-")]
    L_SEP = args.L_sep
    LX = nx - 1
    LY = ny - 1
    LZ = nz - 1
    L_c = LX * scaling[0] + L_SEP
    n_sep = int(np.ceil(L_SEP/scaling[0]))
    voids_output_meshfile = os.path.join(workdir, "voids-unscaled.stl")
    sse_output_meshfile = os.path.join(workdir, "sse-unscaled.stl")
    cam_output_meshfile = os.path.join(workdir, "cam-unscaled.stl")

    scaled_voids_output_meshfile = os.path.join(workdir, "voids.stl")
    scaled_sse_output_meshfile = os.path.join(workdir, "sse.stl")
    scaled_cam_output_meshfile = os.path.join(workdir, "cam.stl")

    print("Processing segmented images")
    for phase in ["voids", "cam", "sse"]:
        print(f"Processing phase {phase}")
        if phase == "sse":
            img_3d = np.ones((nx+n_sep, ny, nz), dtype=bool)
        else:
            img_3d = np.zeros((nx+n_sep, ny, nz), dtype=bool)

        for idx in range(1, nz+1):
            img_file = os.path.join(cam_dir, f"{str(idx).zfill(3)}.tif")
            voids_img_file = os.path.join(voids_dir, f"{str(idx).zfill(3)}.tif")
            cam_img = plt.imread(img_file).copy()[:nx, :ny]
            cam_img[:10, ] = 2
            voids_img = plt.imread(voids_img_file).copy()[:nx, :ny]
            cam_img = np.array(cam_img)
            voids_img = np.array(voids_img)
            if phase == "sse":
                img_3d[:nx, :ny, idx-1] = np.logical_not(np.logical_or(np.isclose(voids_img[:, :], 1), np.isclose(cam_img[:, :], 2)))
            if phase == "voids":
                img_3d[:nx, :ny, idx-1] = np.isclose(voids_img[:, :], 1)
            if phase == "cam":
                img_3d[:nx, :ny, idx-1] = np.logical_and(np.isclose(voids_img[:nx, :ny], 0), np.isclose(cam_img[:, :], 2))
        if phase == "cam":
            img_3d[:10, :, :] = 1
        else:
            img_3d[:10, :, :] = 0
        print(f"Rough {phase} volume fraction {np.average(img_3d[:nx, :ny, :])}")
        mesh = generate_surface_mesh_for_phase(img_3d, (nx+n_sep, ny, nz))
        spacing = [0.5*scaling[0]/L_c, 0.5*scaling[1]/L_c, 0.5*scaling[2]/L_c]
        print(np.unique(mesh.vertices[:, 0]), np.unique(mesh.vertices[:, 1]), np.unique(mesh.vertices[:, 2]))
        scaled_verts = np.vstack((mesh.vertices[:, 0] * spacing[0], mesh.vertices[:, 1] * spacing[1], mesh.vertices[:, 2] * spacing[2]))
        scaled_mesh = trimesh.Trimesh(vertices=scaled_verts.T, faces=mesh.faces)

        if phase == "voids":
            trimesh.exchange.export.export_mesh(mesh, voids_output_meshfile)
            scaled_mesh.export(scaled_voids_output_meshfile)
        elif phase == "sse":
            trimesh.exchange.export.export_mesh(mesh, sse_output_meshfile)
            scaled_mesh.export(scaled_sse_output_meshfile)
        elif phase == "cam":
            trimesh.exchange.export.export_mesh(mesh, cam_output_meshfile)
            scaled_mesh.export(scaled_cam_output_meshfile)
