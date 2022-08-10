#! /usr/bin/env python3

import os
import subprocess
import time

import argparse
import gmsh
import logging
import meshio
import numpy as np

from stl import mesh

import connected_pieces, geometry, utils


FORMAT = '%(asctime)s: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__file__)
logger.setLevel('INFO')


def filter_voxels(voxels, threshold=1):
    new_voxels = voxels
    for coord, v in np.ndenumerate(voxels):
        if v == 0:
            x0, y0, z0 = coord
            try:
                section = voxels[x0 - threshold:x0 + threshold + 1, y0 - threshold : y0 + threshold + 1, z0 - threshold : z0 + threshold + 1]
                total = np.sum(section)
                if total >= (2 * threshold + 1) ** 3 - 2:
                    new_voxels[coord] = 1
            except IndexError:
                continue
    return new_voxels


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
    parser.add_argument("--phase", nargs='?', const=1, default="electrolyte")
    start_time = time.time()
    args = parser.parse_args()
    phase = args.phase
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
    img_dir = os.path.join(args.img_folder, phase)
    mesh_dir = f"mesh/{phase}/{grid_info}_{origin_str}"
    utils.make_dir_if_missing(mesh_dir)
    old_im_files = sorted([os.path.join(img_dir, f) for
                       f in os.listdir(img_dir) if f.endswith(".bmp")])
    im_files = [""] * len(old_im_files)
    for i, f in enumerate(old_im_files):
        fdir = os.path.dirname(f)
        idx = int(f.split("/")[-1].split(".")[0].strip("SegIm"))
        im_files[idx - 3] = f
    n_files = len(im_files)

    start_time = time.time()

    voxels = geometry.load_images_to_voxel(im_files, x_lims=(0, Nx),
                                         y_lims=(0, Ny), z_lims=(0, Nz), origin=origin)
    logger.info("Rough porosity : {:0.4f}".format(np.sum(voxels) / (Lx * Ly * Lz)))
    n_tetrahedra = 0
    n_triangles = 0
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
    stlfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous.stl"
    mshfile = f"mesh/{phase}/{grid_info}_{origin_str}/porous.msh"
    with open(nodefile, "w") as fp:
        fp.write("# node count, 3 dim, no attribute, no boundary marker\n")
        fp.write("%d 3 0 0\n" % int(np.sum(voxels)))
        fp.write("# Node index, node coordinates\n")
        for point_id in range(np.sum(voxels)):
            x, y, z = points_view[point_id]
            fp.write(f"{point_id} {x} {y} {z}\n")

    with open(tetfile, "w") as fp:
        fp.write(f"{n_tetrahedra} 4 0\n")
        for tetrahedron, tet_id in tetrahedra.items():
            p1, p2, p3, p4 = tetrahedron
            fp.write(f"{tet_id} {p1} {p2} {p3} {p4}\n")

    retcode_tetgen = subprocess.check_call(f"tetgen {tetfile} -BkQr", shell=True)
    # generate stl file
    cmd_output = subprocess.check_call(f"./porous.py {mesh_dir} {grid_info}", shell=True)
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", args.resolution)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
    gmsh.model.add("porous")
    gmsh.merge(vtkfile)
    gmsh.merge(stlfile.strip(".stl") + "_left_cc.stl")
    gmsh.merge(stlfile.strip(".stl") + "_right_cc.stl")
    gmsh.merge(stlfile.strip(".stl") + "_insulated.stl")
    gmsh.model.geo.synchronize()
    counter = 0
    volumes = gmsh.model.getEntities(dim=3)
    for i, volume in enumerate(volumes):
        marker = int(counter + i)
        gmsh.model.addPhysicalGroup(volume[0], [volume[1]], marker)
        gmsh.model.setPhysicalName(volume[0], marker, f"VOLUME-{marker}")
    
    surfaces = gmsh.model.getEntities(dim=2)
    print(surfaces)
    insulated = []
    left_cc = []
    right_cc = []
    for i, surface in enumerate(surfaces):
        marker = int(counter + i)
        surf = gmsh.model.addPhysicalGroup(surface[0], [surface[1]], marker)
        gmsh.model.setPhysicalName(surface[0], surf, f"SURFACE-{marker}")
    # for surface in surfaces:
    #     com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
    #     if np.isclose(com[1], 0):
    #         left_cc.append(surface[1])
    #     elif np.isclose(com[1], Ly):
    #         right_cc.append(surface[1])
    #     else:
    #         insulated.append(surface[1])
    # y0_tag = gmsh.model.addPhysicalGroup(2, left_cc)
    # gmsh.model.setPhysicalName(2, y0_tag, "left_cc")
    # yl_tag = gmsh.model.addPhysicalGroup(2, right_cc)
    # gmsh.model.setPhysicalName(2, yl_tag, "right_cc")
    # insulated = gmsh.model.addPhysicalGroup(2, insulated)
    # gmsh.model.setPhysicalName(2, insulated, "insulated")
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(mshfile)
    gmsh.finalize()

    msh = meshio.read(mshfile)
    tetra_mesh = geometry.create_mesh(msh, "tetra")
    meshio.write(f"mesh/{phase}/{grid_info}_{origin_str}/tetr.xdmf", tetra_mesh)
    tria_mesh = geometry.create_mesh(msh, "triangle")
    meshio.write(f"mesh/{phase}/{grid_info}_{origin_str}/tria.xdmf", tria_mesh)

    logger.info("Took {:,} seconds".format(int(time.time() - start_time)))