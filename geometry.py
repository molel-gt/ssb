#!/usr/bin/env python3

import copy

import meshio
import networkx as nx
import numpy as np
import vtk

import commons, constants


phases = commons.Phases()
surface_tags = commons.SurfaceMarkers()
upper_threshold = 0.95
lower_threshold = 0.05


def create_mesh(mesh, cell_type, prune_z=False):
    """
    """
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points,
                           cells={cell_type: cells},
                           cell_data={"name_to_read": [cell_data]}
                           )
    if prune_z:
        out_mesh.prune_z_0()

    return out_mesh


def build_points(data, dp=0):
    """
    key: (x,y,z) coordinate
    value: point_id
    """
    points = {}
    count = 0
    for idx, v in np.ndenumerate(data):
        if v == 1:
            coord = idx
            if dp > 0:
                coord = (round(coord[0], dp), round(coord[1], dp), round(coord[2], dp))
            points[coord] = count
            count += 1

    return points


def build_graph(points, h=1, dp=0):
    """"""
    G = nx.Graph()
    for v in points.values():
        G.add_node(v)
    for k in points.keys():
        x, y, z = k
        if dp == 0:
            neighbors = [
                (int(x + 1), y, z),
                (int(x - 1), y, z),
                (x, int(y + 1), z),
                (x, int(y - 1), z),
                (x, y, int(z + 1)),
                (x, y, int(z - 1)),
            ]
        else:
            neighbors = [
                (round(x + h, dp), y, z),
                (round(x - h, dp), y, z),
                (x, round(y + h, dp), z),
                (x, round(y - h, dp), z),
                (x, y, round(z + h, dp)),
                (x, y, round(z - h, dp)),
            ]
        p0 = points[k]
        for neighbor in neighbors:
            p = points.get(neighbor)
            if p is None:
                continue
            G.add_edge(p0, p)

    return G


def add_boundary_points(points, x_max=50, y_max=50, z_max=50, h=0.5, dp=1):
    """
    A thickness of *h* pixels around the points of one phase to ensure continuity between phases.
    """
    new_points = copy.deepcopy(points)
    max_id = max(new_points.values())
    for (x0, y0, z0), _ in points.items():
        for sign_x in [-1, 0, 1]:
            for sign_y in [-1, 0, 1]:
                for sign_z in [-1, 0, 1]:
                    coord = (round(x0 + h * sign_x, dp), round(y0 + h * sign_y, dp), round(z0 + h * sign_z, dp))
                    if coord[0] > x_max or coord[1] > y_max or coord[2] > z_max:
                        continue
                    if np.less(coord, 0).any():
                        continue
                    v = new_points.get(coord)
                    if v is None:
                        max_id += 1
                        new_points[coord] = max_id

    return new_points


def electrolyte_bordering_active_material(voxels, dp=0):
    effective_electrolyte = set()
    for idx in np.argwhere(voxels == phases.electrolyte):
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
                if value == phases.active_material:
                    if dp > 0:
                        effective_electrolyte.add(tuple([round(v, dp) for v in idx]))
                    else:
                        effective_electrolyte.add(tuple(idx))
            except IndexError:
                continue

    return effective_electrolyte


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


def build_tetrahedra(cubes, points, points_view):
    """
    Build the 5 tetrahedra in a cube
    """
    cubes = build_variable_size_cubes(points, h=0.5)
    tetrahedra = {}
    n_tetrahedra = 0

    for cube in cubes:
        _tetrahedra = []
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
                _tetrahedra.append(tuple(new_tet))
        for tet in _tetrahedra:
            tetrahedra[tet] = n_tetrahedra
            n_tetrahedra += 1

    tets = np.zeros((n_tetrahedra, 4))
    for i, tet in enumerate(list(tetrahedra.keys())):
        tets[i, :] = tet

    return tets


def label_surface_mesh(mesh, effective_electrolyte, transport_length, axis=1):
    """"""
    cells = mesh.get_cells_type("triangle")
    points = mesh.points
    cell_data = np.zeros((cells.shape[0], 2), dtype=np.int32)
    for i, cell in enumerate(cells):
        coords = [tuple(points[j, :]) for j in cell]
        if np.isclose([v[axis] for v in coords], 0).all():
            cell_data[i, 0] = surface_tags.left_cc
        elif np.isclose([v[axis] for v in coords], transport_length).all():
            cell_data[i, 0] = surface_tags.right_cc
        else:
            cell_data[i, 0] = surface_tags.insulated
        if set(coords).issubset(effective_electrolyte):
            cell_data[i, 1] = surface_tags.active
        else:
            cell_data[i, 1] = surface_tags.inactive
        
    return meshio.Mesh(points=points, cells={"triangle": cells}, cell_data={"name_to_read": [cell_data]})


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


def generate_surface_mesh(triangles, effective_electrolyte, shape, points):
    """"""
    _, Ny, _ = shape
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
