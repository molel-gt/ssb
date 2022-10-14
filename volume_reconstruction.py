#! /usr/bin/env python3
import copy
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
import vtk

from skimage import io

import clusters, constants, configs, filter_voxels, geometry, utils


FORMAT = '%(asctime)s: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__file__)
logger.setLevel(configs.get_configs()['LOGGING']['level'])

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


def electrolyte_bordering_active_material(voxels, dp=0):
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
                    if dp > 0:
                        effective_electrolyte.add(tuple([round(v, dp) for v in idx]))
                    else:
                        effective_electrolyte.add(tuple(idx))
            except IndexError:
                continue

    return effective_electrolyte


def generate_surface_mesh(triangles, effective_electrolyte, shape, points):
    """"""
    _, Ny, _ = shape
    prev_points = points
    cells = np.zeros((len(triangles), 3), dtype=np.int32)
    cell_data = np.zeros((cells.shape[0], 2))
    
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

    new_points = np.zeros((len(points.values()) + 1, 3))
    for k, v in points.items():
        new_points[v, :] = k
    points = {}
    points_counter = 0
    for triangle in triangles:
        coord0, coord1, coord2 = [tuple([xv for xv in v]) for v in triangle]
        for coord in triangle:
            if points.get(tuple(coord)) is None:
                points[tuple(coord)] = points_counter
                points_counter += 1

        p0 = points[coord0]
        p1 = points[coord1]
        p2 = points[coord2]
        cells[counter, :] = [p0, p1, p2]
        tags = []
        y_vals = [v[1] for v in triangle]
        if np.isclose(y_vals, 0).all():
            tags.append(surface_tags["left_cc"])
        elif np.isclose(y_vals, Ny - 1).all():
            tags.append(surface_tags["right_cc"])
        else:
            tags.append(surface_tags["insulated"])
        if {coord0, coord1, coord2}.issubset(effective_electrolyte):
            tags.append(surface_tags["active_area"])
        else:
            tags.append(surface_tags["inactive_area"])
        cell_data[counter, :] = tags
        counter += 1
    out_mesh = meshio.Mesh(points=new_points,
                           cells={"triangle": cells},
                           cell_data={"name_to_read": [cell_data]}
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


def build_variable_size_cubes(points, h=0.5):
    """
    Filter out vertices that are malformed/ not part of solid inside or solid surface.
    """
    cubes = []
    for coord, _ in points.items():
        x0, y0, z0 = coord
        face_1 = [(x0, y0, z0), (x0 + h, y0, z0), (x0 + h, y0 + h, z0), (x0, y0 + h, z0)]
        face_2 = [(x0, y0, z0 + h), (x0 + h, y0, z0 + h), (x0 + h, y0 + h, z0 + h), (x0, y0 + h, z0 + h)]
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


def label_surface_mesh(mesh, effective_electrolyte, transport_length, axis=1):
    """"""
    cells = mesh.get_cells_type("triangle")
    points = mesh.points
    cell_data = np.zeros((cells.shape[0], 2), dtype=np.int32)
    for i, cell in enumerate(cells):
        coords = [tuple(points[j, :]) for j in cell]
        if np.isclose([v[axis] for v in coords], 0).all():
            cell_data[i, 0] = surface_tags["left_cc"]
        elif np.isclose([v[axis] for v in coords], transport_length).all():
            cell_data[i, 0] = surface_tags["right_cc"]
        else:
            cell_data[i, 0] = surface_tags["insulated"]
        if set(coords).issubset(effective_electrolyte):
            cell_data[i, 1] = surface_tags["active_area"]
        else:
            cell_data[i, 1] = surface_tags["inactive_area"]
        
    return meshio.Mesh(points=points, cells={"triangle": cells}, cell_data={"name_to_read": [cell_data]})


def extend_points(points, points_master, x_max=50, y_max=50, z_max=50, h=0.5, dp=1):
    """
    A thickness of *h* pixels around the points of one phase to ensure continuity between phases.
    """
    if not isinstance(points, set) or not isinstance(points_master, dict):
        raise TypeError("Accepts points set and points_master dictionary")
    new_points = copy.deepcopy(points)
    for (x0, y0, z0) in points:
        for sign_x in [-1, 0, 1]:
            for sign_y in [-1, 0, 1]:
                for sign_z in [-1, 0, 1]:
                    coord = (round(x0 + h * sign_x, dp), round(y0 + h * sign_y, dp), round(z0 + h * sign_z, dp))
                    if coord[0] > x_max or coord[1] > y_max or coord[2] > z_max:
                        continue
                    if np.less(coord, 0).any():
                        continue
                    v = points_master.get(coord)
                    if v is None:
                        continue
                    new_points.add(coord)
    
    return new_points


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

    points = clusters.build_points(voxels, dp=1)
    points = clusters.add_boundary_points(points, x_max=Nx-1, y_max=Ny-1, z_max=Nz-1, h=0.5, dp=1)
    points_view = {v: k for k, v in points.items()}

    neighbors = number_of_neighbors(voxels)
    effective_electrolyte = electrolyte_bordering_active_material(voxels_filtered, dp=1)
    effective_electrolyte = extend_points(effective_electrolyte, points, x_max=Nx-1, y_max=Ny-1, z_max=Nz-1, h=0.5, dp=1)
    eff_electrolyte_filepath = os.path.join(mesh_dir, "effective_electrolyte.pickle")
    with open(eff_electrolyte_filepath, "wb") as fp:
        pickle.dump(effective_electrolyte, fp, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Rough porosity : {:0.4f}".format(np.sum(voxels) / (Nx * Ny * Nz)))
    # Only label points that will be used for meshing
    n_tetrahedra = 0
    n_triangles = 0
    counter = 0
    tetrahedra = {}
    
    cubes = build_variable_size_cubes(points, h=0.5)
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
    tetr_mesh_unscaled = geometry.create_mesh(tet_msh, "tetra")
    tetr_mesh_unscaled.write(tetr_xdmf_unscaled)
    tetr_mesh_scaled = scale_mesh(tetr_mesh_unscaled, "tetra", scale_factor=scale_factor)
    tetr_mesh_scaled.write(tetr_xdmf_scaled)

    retcode_paraview = subprocess.check_call("pvpython extract_surface_from_volume.py {}".format(os.path.dirname(tetr_xdmf_unscaled)), shell=True)
    surf_msh = meshio.read(tria_xmf_unscaled)
    tria_mesh_unscaled = label_surface_mesh(surf_msh, effective_electrolyte, Ny - 1)
    tria_mesh_unscaled.write(tria_xdmf_unscaled)

    tria_mesh_scaled = scale_mesh(tria_mesh_unscaled, "triangle", scale_factor=scale_factor)
    tria_mesh_scaled.write(tria_xdmf_scaled)
    
    logger.info("Took {:,} seconds".format(int(time.time() - start_time)))