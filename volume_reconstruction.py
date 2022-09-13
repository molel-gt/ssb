#! /usr/bin/env python3

from bdb import effective
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

import connected_pieces, filter_voxels, geometry, utils


FORMAT = '%(asctime)s: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__file__)
logger.setLevel('INFO')

upper_threshold = 0.95
lower_threshold = 0.05

phase_key = {
    "void": 0,
    "activematerial": 1,
    "electrolyte": 2,
}

surface_tags = {
    "left_cc": 0,
    "right_cc": 1,
    "insulated": 2,
    "inactive_area": 3,
    "active_area": 4,
}

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
        np_pts = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())])
        triangles.append(np_pts)
    
    return triangles


def electrolyte_bordering_active_material(voxels):
    effective_electrolyte = set()
    for idx in np.argwhere(voxels == phase_key["electrolyte"]):
        x, y, z = idx
        neighbors = [
            (x, y + 1, z),
            (x, y - 1, z),
            (x + 1, y, z),
            (x - 1, y, z),
            (x, y, z + 1),
            (x, y, z - 1),
        ]
        for p in neighbors:
            try:
                value = voxels[p]
            except IndexError:
                continue
            if value == phase_key["activematerial"]:
                effective_electrolyte.add(set(p))
    
    return effective_electrolyte


def generate_surface_mesh(triangles, effective_electrolyte, points, shape):
    """"""
    _, Ny, _ = shape
    cells = np.zeros((len(triangles), 3))
    cell_data = []
    point_ids = set()
    new_points = np.zeros((max(points.values()) + 1, 3))
    for k, v in points.items():
        new_points[v, :] = k
    counter = 0
    for triangle in triangles:
        coord0, coord1, coord2 = [tuple(v) for v in triangle]
        try:
            p0 = points[coord0]
            p1 = points[coord1]
            p2 = points[coord2]
        except KeyError:
            continue
        cells[counter, :] = [p0, p1, p2]
        counter += 1
        point_ids |= {p0, p1, p2}
        tags = []
        y_vals = [v[1] for v in triangle]
        if np.isclose(y_vals, 0).all():
            tags.append(surface_tags["left_cc"])
        elif np.isclose(y_vals, Ny - 1).all():
            tags.append(surface_tags["right_cc"])
        else:
            tags.append(surface_tags["insulated"])
        if {p0, p1, p2}.issubset(effective_electrolyte):
            tags.append(surface_tags["active_area"])
        else:
            tags.append(surface_tags["inactive_area"])
        cell_data.append(tags)
    out_mesh = meshio.Mesh(points=new_points,
                           cells={"triangle": cells},
                           cell_data={"name_to_read": [cell_data]}
                           )
    return out_mesh


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='computes specific area')
    parser.add_argument('--img_folder', help='bmp files parent directory',
                        required=True)
    parser.add_argument('--grid_info', help='Nx-Ny-Nz',
                        required=True)
    parser.add_argument('--origin', default=(0, 0, 0), help='where to extract grid from')
    parser.add_argument("--resolution", nargs='?', const=1, default=0.5, type=float)
    parser.add_argument("--phase", nargs='?', const=1, default=0, type=int)
    parser.add_argument("--scale_x", nargs='?', const=1, default=1, type=lambda f: np.around(float(f), 8))
    parser.add_argument("--scale_y", nargs='?', const=1, default=1, type=lambda f: np.around(float(f), 8))
    parser.add_argument("--scale_z", nargs='?', const=1, default=1, type=lambda f: np.around(float(f), 8))
    start_time = time.time()
    args = parser.parse_args()
    phase = args.phase
    scale_x = args.scale_x
    scale_y = args.scale_y
    scale_z = args.scale_z
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
    img_dir = os.path.join(args.img_folder) #, str(phase))
    mesh_dir = f"mesh/{phase}/{grid_info}_{origin_str}"
    utils.make_dir_if_missing(mesh_dir)
    im_files = sorted([os.path.join(args.img_folder, f) for
                       f in os.listdir(args.img_folder) if f.endswith(".tif")])
    n_files = len(im_files)

    start_time = time.time()

    shape = [*io.imread(im_files[0]).shape, n_files]
    voxels_all = filter_voxels.load_images(im_files, shape)[:Nx, :Ny, :Nz]
    filtered = filter_voxels.get_filtered_voxels(voxels_all)
    voxels = np.isclose(filtered, phase)
    logger.info("Rough porosity : {:0.4f}".format(np.sum(voxels) / (Lx * Ly * Lz)))
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
    nodefile = f"mesh/{phase}/{grid_info}_{origin_str}/porous.node"
    tetfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous.ele"
    facesfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous.face"
    vtkfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous.1.vtk"
    surface_vtk = f"mesh/{phase}/{grid_info}_{origin_str}/surface.vtk"
    stlfile = f"mesh/{phase}/{grid_info}_{origin_str}/new-porous.stl"
    mshfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous_tetr.msh"
    tetr_xdmf = f"mesh/{phase}/{grid_info}_{origin_str}/tetr.xdmf"
    tria_xdmf = f"mesh/{phase}/{grid_info}_{origin_str}/tria.xdmf"
    with open(nodefile, "w") as fp:
        fp.write("# node count, 3 dim, no attribute, no boundary marker\n")
        fp.write("%d 3 0 0\n" % int(np.sum(voxels)))
        fp.write("# Node index, node coordinates\n")
        for point_id in range(np.sum(voxels)):
            x0, y0, z0 = points_view[point_id]
            x = np.around(scale_x * x0, 8)
            y = np.around(scale_y * y0, 8)
            z = np.around(scale_x * z0, 8)
            fp.write(f"{point_id} {x} {y} {z}\n")

    with open(tetfile, "w") as fp:
        fp.write(f"{n_tetrahedra} 4 0\n")
        for tetrahedron, tet_id in tetrahedra.items():
            p1, p2, p3, p4 = tetrahedron
            fp.write(f"{tet_id} {p1} {p2} {p3} {p4}\n")

    retcode_tetgen = subprocess.check_call(f"tetgen {tetfile} -BdkQr", shell=True)
    retcode_paraview = subprocess.check_call("pvpython extractSurf.py {}".format(os.path.dirname(surface_vtk)), shell=True)
    triangles = read_vtk_surface(surface_vtk)
    effective_electrolyte = electrolyte_bordering_active_material(voxels)
    tria_mesh = generate_surface_mesh(triangles, effective_electrolyte, points, shape)
    meshio.write(tria_xdmf, tria_mesh)

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", args.resolution)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
    gmsh.model.add("porous")
    gmsh.merge(vtkfile)
    gmsh.model.occ.synchronize()

    volumes = gmsh.model.getEntities(dim=3)
    for i, volume in enumerate(volumes):
        marker = int(counter + i)
        gmsh.model.addPhysicalGroup(3, [volume[1]], marker)
        gmsh.model.setPhysicalName(3, marker, f"V{marker}")
    gmsh.model.occ.synchronize()
    
    insulated = []
    left_cc = []
    right_cc = []
    surfaces = gmsh.model.getEntities(dim=2)
    for surface in surfaces:
        surf = gmsh.model.addPhysicalGroup(2, [surface[1]])
        gmsh.model.setPhysicalName(2, surf, f"S{surf}")
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(mshfile)
    gmsh.finalize()

    vol_msh = meshio.read(mshfile)
    tetra_mesh = geometry.create_mesh(vol_msh, "tetra")
    meshio.write(tetr_xdmf, tetra_mesh)

    logger.info("Took {:,} seconds".format(int(time.time() - start_time)))