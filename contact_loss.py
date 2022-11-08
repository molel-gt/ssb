#!/usr/bin/env python3

import os
import pickle
import subprocess

import argparse
import gmsh
import meshio
import numpy as np

from skimage import io

import commons, configs, geometry, utils

markers = commons.SurfaceMarkers()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Volume with Contact Loss")
    parser.add_argument("--grid_info", help="Nx-Ny-Nz that defines the grid size", required=True)
    parser.add_argument("--contact_map", help="Image to generate contact map", required=True)
    parser.add_argument("--phase", help="0 -> void, 1 -> SE, 2 -> AM", nargs='?', const=1, default=1)
    args = parser.parse_args()
    grid_info = args.grid_info
    contact_img_file = args.contact_map
    phase = args.phase
    scaling = configs.get_configs()['VOXEL_SCALING']
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])
    scale_factor = (scale_x, scale_y, scale_z)
    dp = int(configs.get_configs()['GEOMETRY']['dp'])
    h = float(configs.get_configs()['GEOMETRY']['h'])
    origin_str = 'contact_loss'
    mesh_dir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], 'contact_loss', grid_info)
    Nx, Ny, Nz = [int(v) for v in grid_info.split("-")]
    Lx = Nx - 1
    Ly = Ny - 1
    Lz = Nz - 1
    utils.make_dir_if_missing(mesh_dir)
    contact_points_filepath = os.path.join(mesh_dir, "contact_points.pickle")
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

    img = io.imread(contact_img_file)
    contact_points = set()
    for idx in np.argwhere(np.isclose(img, phase)):
        contact_points.add(tuple([int(v) for v in idx] + [0]))

    with open(contact_points_filepath, "wb") as fp:
        pickle.dump(contact_points, fp, protocol=pickle.HIGHEST_PROTOCOL)

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
    gmsh.model.add("porous")
    circle = gmsh.model.occ.addCircle(12.5, 12.5, 0, 7)
    gmsh.model.occ.synchronize()
    box = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
    gmsh.model.occ.synchronize()
    grp1 = gmsh.model.addPhysicalGroup(3, [box], args.phase)
    gmsh.model.setPhysicalName(3, grp1, "conductor")
    gmsh.model.occ.synchronize()
    # circle = gmsh.model.occ.addCircle(12.5, 12.5, 0, 7)
    
    grp2 = gmsh.model.addPhysicalGroup(2, [circle], markers.left_cc)
    gmsh.model.setPhysicalName(2, grp2, "left_cc")
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.occ.getEntities(dim=2)
    walls = []
    for surface in surfaces:
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        if np.isclose(com[2], Lz):
            right_cc = gmsh.model.addPhysicalGroup(2, [surface[1]])
            gmsh.model.setPhysicalName(2, markers.right_cc, "right_cc")

    print(circle, grp1)

    # points = [
    #     (0, 0, 0),
    #     (Lx, 0, 0),
    #     (Lx, Ly, 0),
    #     (0, Ly, 0),
    #     (0, 0, Lz),
    #     (Lx, 0, Lz),
    #     (Lx, Ly, Lz),
    #     (0, Ly, Lz),
    # ]
    # gmsh_points = [
    # ]

    # for point in points:
    #     gmsh_points.append(gmsh.model.occ.addPoint(*point))
    
    # lines = [
    #     gmsh.model.occ.addLine(gmsh_points[0], gmsh_points[1]),
    #     gmsh.model.occ.addLine(gmsh_points[1], gmsh_points[2]),
    #     gmsh.model.occ.addLine(gmsh_points[2], gmsh_points[3]),
    #     gmsh.model.occ.addLine(gmsh_points[3], gmsh_points[0]),
    #     gmsh.model.occ.addLine(gmsh_points[4], gmsh_points[5]),
    #     gmsh.model.occ.addLine(gmsh_points[5], gmsh_points[6]),
    #     gmsh.model.occ.addLine(gmsh_points[6], gmsh_points[7]),
    #     gmsh.model.occ.addLine(gmsh_points[7], gmsh_points[4]),
    #     gmsh.model.occ.addLine(gmsh_points[5], gmsh_points[1]),
    #     gmsh.model.occ.addLine(gmsh_points[2], gmsh_points[6]),
    #     gmsh.model.occ.addLine(gmsh_points[0], gmsh_points[4]),
    #     gmsh.model.occ.addLine(gmsh_points[3], gmsh_points[7]),
    #     ]
    
    # loops = [
    #     gmsh.model.occ.addCurveLoop(lines[:4]),
    #     gmsh.model.occ.addCurveLoop([lines[1], lines[8], lines[5], lines[9]]),
    #     gmsh.model.occ.addCurveLoop(lines[4:8]),
    #     gmsh.model.occ.addCurveLoop([lines[3], lines[11], lines[7], lines[10]]),
    #     gmsh.model.occ.addCurveLoop([lines[0], lines[8], lines[4], lines[10]]),
    #     gmsh.model.occ.addCurveLoop([lines[2], lines[9], lines[6], lines[11]]),
    # ]
    # surfaces = []
    # for loop in loops:
    #     surfaces.append(
    #         gmsh.model.occ.addPlaneSurface((1, loop))
    #     )
    # print(surfaces)
    # # volume = gmsh.model.geo.addSurfaceLoop(surfaces)
    # tag = gmsh.model.geo.addPhysicalGroup(3, [volume], args.phase)
    # gmsh.model.setPhysicalName(3, tag, "conductor")


    # box = np.ones((Nx, Ny, Nz), dtype=np.uint8)
    # points = geometry.build_points(box)
    # points = geometry.add_boundary_points(points, x_max=Lx, y_max=Ly, z_max=Lz, h=h, dp=dp)
    # cubes = geometry.build_variable_size_cubes(points, h=h, dp=dp)
    # tetrahedra = geometry.build_tetrahedra(cubes, points)
    # geometry.write_points_to_node_file(nodefile, points)
    # geometry.write_tetrahedra_to_ele_file(tetfile, tetrahedra)

    # retcode_tetgen = subprocess.check_call(f"tetgen {tetfile} -rkQF", shell=True)
    # gmsh.initialize()
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 2)
    # gmsh.model.add("porous")
    # gmsh.merge(vtkfile)
    # gmsh.model.occ.synchronize()
    # counter = 0
    # volumes = gmsh.model.getEntities(dim=3)
    # surfaces = gmsh.model.getEntities(dim=2)
    # print(surfaces)
    # for i, volume in enumerate(volumes):
    #     marker = int(counter + i)
    #     gmsh.model.addPhysicalGroup(3, [volume[1]], marker)
    #     gmsh.model.setPhysicalName(3, marker, f"V{marker}")
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(tetr_mshfile)
    gmsh.finalize()

    tet_msh = meshio.read(tetr_mshfile)
    tetr_mesh_unscaled = geometry.create_mesh(tet_msh, "tetra")
    tetr_mesh_unscaled.write(tetr_xdmf_unscaled)
    tetr_mesh_scaled = geometry.scale_mesh(tetr_mesh_unscaled, "tetra", scale_factor=scale_factor)
    tetr_mesh_scaled.write(tetr_xdmf_scaled)

    tria_mesh_unscaled = geometry.create_mesh(tet_msh, "triangle")
    tria_mesh_unscaled.write(tria_xdmf_unscaled)
    tria_mesh_scaled = geometry.scale_mesh(tria_mesh_unscaled, "triangle", scale_factor=scale_factor)
    tria_mesh_scaled.write(tria_xdmf_scaled)
