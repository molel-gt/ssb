#! /usr/bin/env python3

import gc
import os
import subprocess
import time

import argparse
import gmsh
import logging
import matplotlib.pyplot as plt
import meshio
import numpy as np
import vtk

from skimage import io

import connected_pieces, constants, filter_voxels, geometry, utils


FORMAT = '%(asctime)s: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__file__)
logger.setLevel('INFO')

upper_threshold = 0.95
lower_threshold = 0.05

phase_key = constants.phase_key
surface_tags = constants.surface_tags


def number_of_neighbors(voxels):
    """"""
    num_neighbors = np.zeros(voxels.shape, dtype=np.uint8)
    for idx in np.argwhere(voxels == True):
        x, y, z = idx
        neighbors = [
            (x, y + 1, z),
            (x, y - 1, z),
            (x + 1, y, z),
            (x - 1, y, z),
            (x, y, z + 1),
            (x, y, z - 1),
        ]
        sum_neighbors = 0
        for p in neighbors:
            try:
                phase_value = voxels[p]
                sum_neighbors += phase_value
            except IndexError:
                continue
        num_neighbors[(x, y, z)] = sum_neighbors
    
    return num_neighbors


def marching_cubes_filter(voxel_cube):
    """"""
    nx, ny, nz = voxel_cube.shape
    max_volume = nx ** 3
    max_area = nx ** 2
    total = np.sum(voxel_cube)
    if np.isclose(total, max_volume):
        return voxel_cube
    if total < 3: #lower_threshold * max_volume:
        return np.zeros(voxel_cube.shape, dtype=voxel_cube.dtype)
    if total >= upper_threshold * max_volume:
        avg_x0 = np.sum(voxel_cube[0, :, :])
        avg_xL = np.sum(voxel_cube[-1, :, :])
        avg_y0 = np.sum(voxel_cube[:, 0, :])
        avg_yL = np.sum(voxel_cube[:, -1, :])
        avg_z0 = np.sum(voxel_cube[:, :, 0])
        avg_zL = np.sum(voxel_cube[:, :, -1])
        if np.isclose(np.array([avg_x0, avg_xL, avg_y0, avg_yL, avg_z0, avg_zL]), max_area).all():
            return np.ones(voxel_cube.shape, voxel_cube.dtype)

    return voxel_cube


def apply_filter_to_3d_array(array, size):
    nx, ny, nz = array.shape
    filtered_array = np.zeros(array.shape, array.dtype)
    x_indices = range(0, nx, size)
    y_indices = range(0, ny, size)
    z_indices = range(0, nz, size)
    for idx in x_indices[:-1]:
        for idy in y_indices[:-1]:
            for idz in z_indices[:-1]:
                upper_x = idx + size if (idx + size) < nx else -1
                upper_y = idy + size if (idy + size) < ny else -1
                upper_z = idz + size if (idz + size) < nz else -1
                chunk = array[idx:upper_x, idy:upper_y, idz:upper_z]
                filtered_array[idx:upper_x, idy:upper_y, idz:upper_z] = marching_cubes_filter(chunk)

    return filtered_array


def read_vtk_surface(file_path):
    """"""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()

    polydata = reader.GetOutput()
    triangles = []

    for i in range(polydata.GetNumberOfCells()):
        pts = polydata.GetCell(i).GetPoints()    
        np_pts = np.array([pts.GetPoint(j) for j in range(pts.GetNumberOfPoints())], dtype=np.int32)
        triangles.append(np_pts)
    
    return triangles


def electrolyte_bordering_active_material(voxels):
    effective_electrolyte = set()
    for idx in np.argwhere(voxels == phase_key["electrolyte"]):
        x, y, z = [int(v) for v in idx]
        neighbors = [
            (x, int(y + 1), z),
            (x, int(y - 1), z),
            (int(x + 1), y, z),
            (int(x - 1), y, z),
            (x, y, int(z + 1)),
            (x, y, int(z - 1)),
        ]
        for p in neighbors:
            try:
                value = voxels[p]
                if value == phase_key["activematerial"]:
                    effective_electrolyte.add(tuple(idx))
            except IndexError:
                continue

    return effective_electrolyte


def generate_surface_mesh(triangles, effective_electrolyte, points, shape, scaling=(1, 1, 1)):
    """"""
    _, Ny, _ = shape
    cells = np.zeros((len(triangles), 3), dtype=np.int32)
    cell_data = []
    scale_y = scaling[1]
    
    counter = 0
    points_counter = 0
    points0 = {}
    for triangle in triangles:
        coord0, coord1, coord2 = [tuple(v) for v in triangle]
        for coord in triangle:
            if points0.get(tuple(coord)) is None:
                points0[tuple(coord)] = points_counter
                points_counter += 1
    points = points0

    new_points = np.zeros((max(points.values()) + 1, 3))
    for k, v in points.items():
        new_points[v, :] = k

    for triangle in triangles:
        coord0, coord1, coord2 = [tuple([xv for xv in v]) for v in triangle]

        p0 = points[coord0]
        p1 = points[coord1]
        p2 = points[coord2]
        cells[counter, :] = [p0, p1, p2]
        counter += 1
        tags = []
        y_vals = [v[1] for v in triangle]
        if np.isclose(y_vals, 0).all():
            tags.append(surface_tags["left_cc"])
        elif np.isclose(y_vals, np.around(scale_y * (Ny - 1), 8)).all():
            tags.append(surface_tags["right_cc"])
        else:
            tags.append(surface_tags["insulated"])
        coord0 = (int(np.rint(coord0[0]/scale_x)), int(np.rint(coord0[1]/scale_y)), int(np.rint(coord0[2]/scale_z)))
        coord1 = (int(np.rint(coord1[0]/scale_x)), int(np.rint(coord1[1]/scale_y)), int(np.rint(coord1[2]/scale_z)))
        coord2 = (int(np.rint(coord2[0]/scale_x)), int(np.rint(coord2[1]/scale_y)), int(np.rint(coord2[2]/scale_z)))
        if {coord0, coord1, coord2}.issubset(effective_electrolyte):
            tags.append(surface_tags["active_area"])
        else:
            tags.append(surface_tags["inactive_area"])
        cell_data.append(tags)
    out_mesh = meshio.Mesh(points=new_points,
                           cells={"triangle": cells},
                           cell_data={"name_to_read": [np.array([v for v in cell_data])]}
                           )
    return points, out_mesh


def build_cubes(voxels, points):
    """
    Filter out vertices that are malformed/ not part of solid inside or solid surface.
    """
    cubes = []
    for idx in np.argwhere(voxels == 1):
        x0, y0, z0 = idx
        face_1 = [(x0, y0, z0), (x0 + 1, y0, z0), (x0 + 1, y0 + 1, z0), (x0, y0 + 1, z0)]
        face_2 = [(x0, y0, z0 + 1), (x0 + 1, y0, z0 + 1), (x0 + 1, y0 + 1, z0 + 1), (x0, y0 + 1, z0 + 1)]
        faces = [[], []]
        n_points = 0
        for i in range(4):
            if points.get(face_1[i]) is None:
                break
            faces[0].append(points.get(face_1[i]))
            n_points += 1
            if points.get(face_2[i]) is None:
                break
            n_points += 1
            faces[1].append(points.get(face_2[i]))
        cubes.append((face_1, face_2))
    return cubes


def build_tetrahedra(cube, points):
    """
    Build the 6 tetrahedra in a cube
    """
    tetrahedra = []
    p0, p1, p2, p3 = cube[0]
    p4, p5, p6, p7 = cube[1]
    for i in range(5):
        if i == 0:
            tet = (p0, p1, p3, p4)
        if i == 1:
            tet = (p1, p2, p3, p6)
        if i == 2:
            tet = (p4, p5, p6, p1)
        if i == 3:
            tet = (p4, p7, p6, p3)
        if i == 4:
            tet = (p4, p6, p1, p3)
        tet_ok = True
        new_tet = []
        for p in tet:
            new_p = points.get(p)
            new_tet.append(new_p)
            if new_p is None:
                tet_ok = False
        if tet_ok:
            tetrahedra.append(tuple(new_tet))
    return tetrahedra


def scale_mesh(mesh, cell_type, scale_factor=(1, 1, 1)):
    """"""
    scaled_points = np.zeros(mesh.points.shape, dtype=np.double)

    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("name_to_read", cell_type)
    for idx, point in enumerate(mesh.points):
        point_scaled = tuple(np.format_float_scientific(v, exp_digits=constants.EXP_DIGITS) for v in np.array(point) * np.array(scale_factor))
        scaled_points[idx, :] = point_scaled
    out_mesh = meshio.Mesh(points=scaled_points,
                           cells={cell_type: cells},
                           cell_data={"name_to_read": [cell_data]}
                           )
    return out_mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='computes specific area')
    parser.add_argument('--img_folder', help='bmp files parent directory',
                        required=True)
    parser.add_argument('--grid_info', help='Nx-Ny-Nz',
                        required=True)
    parser.add_argument('--origin', default=(0, 0, 0), help='where to extract grid from')
    parser.add_argument("--resolution", nargs='?', const=1, default=0.5, type=float)
    parser.add_argument("--phase", nargs='?', const=1, default=1, type=int)
    parser.add_argument("--scale_x", nargs='?', const=1, default=1, type=lambda f: np.around(float(f), 8))
    parser.add_argument("--scale_y", nargs='?', const=1, default=1, type=lambda f: np.around(float(f), 8))
    parser.add_argument("--scale_z", nargs='?', const=1, default=1, type=lambda f: np.around(float(f), 8))
    start_time = time.time()
    args = parser.parse_args()
    phase = args.phase
    scale_x = args.scale_x
    scale_y = args.scale_y
    scale_z = args.scale_z
    scale_factor = (scale_x, scale_y, scale_z)
    if isinstance(args.origin, str):
        origin = tuple(map(lambda v: int(v), args.origin.split(",")))
    else:
        origin = args.origin
    origin_str = "-".join([str(v).zfill(3) for v in origin])
    grid_info = "-".join([v.zfill(3) for v in args.grid_info.split("-")])
    grid_size = int(args.grid_info.split("-")[0])
    Nx, Ny, Nz = [int(v) for v in args.grid_info.split("-")]
    Lx = Nx - 1
    Ly = Ny - 1
    Lz = Nz - 1
    img_dir = os.path.join(args.img_folder)
    mesh_dir = f"mesh/{phase}/{grid_info}_{origin_str}"
    utils.make_dir_if_missing(mesh_dir)
    im_files = sorted([os.path.join(args.img_folder, f) for
                       f in os.listdir(args.img_folder) if f.endswith(".tif")])
    n_files = len(im_files)

    start_time = time.time()

    shape = [*io.imread(im_files[0]).shape, n_files]
    voxels_raw = filter_voxels.load_images(im_files, shape)[:Nx, :Ny, :Nz]
    voxels_filtered = filter_voxels.get_filtered_voxels(voxels_raw)
    voxels = np.isclose(voxels_filtered, phase)

    neighbors = number_of_neighbors(voxels)
    effective_electrolyte = electrolyte_bordering_active_material(voxels_filtered)
    logger.info("Rough porosity : {:0.4f}".format(np.sum(voxels) / (Nx * Ny * Nz)))
    # Only label points that will be used for meshing
    n_tetrahedra = 0
    n_triangles = 0
    counter = 0
    tetrahedra = {}
    points = connected_pieces.build_points(voxels)
    points_view = {v: k for k, v in points.items()}
    cubes = build_cubes(voxels, points)
    for cube in cubes:
        _tetrahedra = build_tetrahedra(cube, points)
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
    dummy_voxels = np.zeros(voxels.shape, dtype=voxels.dtype)
    for coord in points_set:
        dummy_voxels[coord] = 1
    new_neighbors = number_of_neighbors(dummy_voxels)
    points = {}
    points_view = {}
    points_id = 0
    for idx in np.argwhere(new_neighbors < 6):
        points[tuple(idx)] = points_id
        points_id += 1
    for point in points_set:
        coord = point
        if points.get(coord) is None:
            points[coord] = points_id
            points_id += 1
    points_view = {v: k for k, v in points.items()}
    tetrahedra = np.zeros((n_tetrahedra, 4))
    for i in range(n_tetrahedra):
        for j, k in enumerate(range(0, 12, 3)):
            tetrahedra[i, j] = int(points[tuple(tetrahedra_np[i, k:k+3])])

    nodefile = f"mesh/{phase}/{grid_info}_{origin_str}/porous.node"
    tetfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous.ele"
    facesfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous.face"
    vtkfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous.1.vtk"
    surface_vtk = f"mesh/{phase}/{grid_info}_{origin_str}/surface.vtk"
    stlfile = f"mesh/{phase}/{grid_info}_{origin_str}/new-porous.stl"
    mshfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous_tetr.msh"
    surf_mshfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous_tria.msh"
    tetr_xdmf_scaled = f"mesh/{phase}/{grid_info}_{origin_str}/tetr.xdmf"
    tetr_xdmf_unscaled = f"mesh/{phase}/{grid_info}_{origin_str}/tetr_unscaled.xdmf"
    tria_xdmf_scaled = f"mesh/{phase}/{grid_info}_{origin_str}/tria.xdmf"
    tria_xdmf_unscaled = f"mesh/{phase}/{grid_info}_{origin_str}/tria_unscaled.xdmf"

    with open(nodefile, "w") as fp:
        fp.write("%d 3 0 1\n" % int(len(points.values())))
        for coord, point_id in points.items():
            x0, y0, z0 = coord

            # set tags
            tag1 = 0
            tag2 = None
            if neighbors[(x0, y0, z0)] < 6:
                if np.isclose(y0, 0):
                    tag1 = surface_tags["left_cc"]
                elif np.isclose(y0, Ny - 1):
                    tag1 = surface_tags["right_cc"]
                else:
                    tag1 = surface_tags["insulated"]
                # if {x0, y0, z0} in effective_electrolyte:
                #     print("IN")
                #     tag2 = surface_tags["active_area"]
                # else:
                #     tag2 = surface_tags["inactive_area"]

            fp.write(f"{point_id} {x0} {y0} {z0} {tag1}\n")

    with open(tetfile, "w") as fp:
        fp.write(f"{n_tetrahedra} 4 0\n")
        for tet_id, tetrahedron in enumerate(tetrahedra):
            p1, p2, p3, p4 = [int(v) for v in tetrahedron]
            fp.write(f"{tet_id} {p1} {p2} {p3} {p4}\n")

    # Free up memory of objects we won't use
    tetrahedra = None
    tetrahedra_np = None
    cubes = None
    voxels = None
    gc.collect()

    retcode_tetgen = subprocess.check_call(f"tetgen {tetfile} -rkQF", shell=True)
    retcode_paraview = subprocess.check_call("pvpython extractSurf.py {}".format(os.path.dirname(surface_vtk)), shell=True)

    # Surface Mesh
    triangles = read_vtk_surface(surface_vtk)
    new_points, tria_mesh_unscaled = generate_surface_mesh(triangles, effective_electrolyte, (Nx, Ny, Nz), points)
    tria_mesh_unscaled.write(tria_xdmf_unscaled)

    vol_msh = meshio.read(vtkfile)
    # msh_points = vol_msh.points
    # old_points = np.zeros(msh_points.shape)
    # points.update(new_points)
    # for k, v in points.items():
    #     old_points[v, :] = k
    # vol_msh.points = old_points
    tetra_mesh_unscaled = geometry.create_mesh(vol_msh, "tetra")
    tetra_mesh_unscaled.write(tetr_xdmf_unscaled)

    # Geometry Scaling
    tetra_mesh_scaled = scale_mesh(tetra_mesh_unscaled, "tetra", scale_factor=scale_factor)
    tetra_mesh_scaled.write(tetr_xdmf_scaled)
    tria_mesh_scaled = scale_mesh(tria_mesh_unscaled, "triangle", scale_factor=scale_factor)
    tria_mesh_scaled.write(tria_xdmf_scaled)

    logger.info("Took {:,} seconds".format(int(time.time() - start_time)))