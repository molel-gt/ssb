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
    parser.add_argument("--eps", help="coverage of area at left cc", nargs='?', const=1, default=0.05)
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
    mesh_dir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], 'contact_loss', grid_info, str(args.eps))
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

    # img = io.imread(contact_img_file)
    # contact_points = set()
    # for idx in np.argwhere(np.isclose(img, phase)):
    #     contact_points.add(tuple([int(v) for v in idx] + [0]))

    # with open(contact_points_filepath, "wb") as fp:
    #     pickle.dump(contact_points, fp, protocol=pickle.HIGHEST_PROTOCOL)
    r = 2 * Lx * (args.eps/np.pi) ** 0.5
    xc, yc = 0.5 * Lx, 0.5 * Ly
    # gmsh.initialize()
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.25)
    # model = gmsh.model.occ
    # points = []
    # points.append(model.addPoint(0, 0, 0))
    # points.append(model.addPoint(Lx, 0, 0))
    # points.append(model.addPoint(Lx, Ly, 0))
    # points.append(model.addPoint(0, Ly, 0))
    # points.append(model.addPoint(0, 0, Lz))
    # points.append(model.addPoint(Lx, 0, Lz))
    # points.append(model.addPoint(Lx, Ly, Lz))
    # points.append(model.addPoint(0, Ly, Lz))

    # points.append(model.addPoint(xc, yc, 0))
    # points.append(model.addPoint(xc + r, yc, 0))
    # points.append(model.addPoint(xc, yc + r, 0))
    # points.append(model.addPoint(xc - r, yc, 0))
    # points.append(model.addPoint(xc, yc - r, 0))

    # lines = []
    # face1 = points[:4]
    # face2 = points[4:8]
    # for i in range(-1, 3):
    #     lines.append(model.addLine(face1[i], face1[i + 1]))
    # for i in range(-1, 3):
    #     lines.append(model.addLine(face2[i], face2[i + 1]))
    # lines.append(model.addLine(points[4], points[0]))
    # lines.append(model.addLine(points[3], points[7]))
    # lines.append(model.addLine(points[1], points[5]))
    # lines.append(model.addLine(points[2], points[6]))
    # arcs = []
    # print()
    # arcs.append(model.addCircleArc(points[9], points[8], points[10]))
    # arcs.append(model.addCircleArc(points[10], points[8], points[11]))
    # arcs.append(model.addCircleArc(points[11], points[8], points[12]))
    # arcs.append(model.addCircleArc(points[12], points[8], points[9]))
    # curve_loops = [""] * 7
    # curve_loops[0] = model.addCurveLoop(arcs)
    # surfaces = [""] * 7
    # print(curve_loops[0])
    # surfaces[0] = model.addPlaneSurface((1, curve_loops[0]))
    # curve_loops[6] = model.addCurveLoop(lines[:4])
    # surfaces[6] = model.addPlaneSurface((1, [curve_loops[6], curve_loops[0]]))
    # curve_loops[1] = model.addCurveLoop(lines[4:8])
    # surfaces[1] = model.addPlaneSurface((1, curve_loops[1]))
    # curve_loops[2] = model.addCurveLoop(lines[3], lines[8], lines[7], lines[9])
    # surfaces[2] = model.addPlaneSurface((1, curve_loops[2]))
    # curve_loops[3] = model.addCurveLoop(lines[1], lines[10], lines[5], lines[11])
    # surfaces[3] = model.addPlaneSurface((1, curve_loops[3]))
    # curve_loops[4] = model.addCurveLoop(lines[2], lines[11], lines[7], lines[9])
    # surfaces[4] = model.addPlaneSurface((1, curve_loops[4]))
    # curve_loops[5] = model.addCurveLoop(lines[0], lines[10], lines[4], lines[8])
    # surfaces[5] = model.addPlaneSurface((1, curve_loops[5]))
    # model.addPhysicalGroup(2, surfaces[0], markers.left_cc)
    # model.addPhysicalGroup(2, surfaces[1], markers.right_cc)
    # model.addPhysical(2, [surfaces[i] for i in [2, 3, 4, 5, 6]], markers.insulated)
    # surfloop = model.addSurfaceLoop(surfaces)
    # vol = model.addVolume((3, surfloop))
    # model.addPhysicalGroup(3, [vol], args.phase)
    # gmsh.model.occ.synchronize()

    # gmsh.model.mesh.generate(3)
    # gmsh.write(tetr_mshfile)
    # gmsh.finalize()
    res = subprocess.check_call(f"gmsh -3 contact-loss.geo -o {tetr_mshfile}", shell=True)
    tet_msh = meshio.read(tetr_mshfile)
    tetr_mesh_unscaled = geometry.create_mesh(tet_msh, "tetra")
    tetr_mesh_unscaled.write(tetr_xdmf_unscaled)
    tetr_mesh_scaled = geometry.scale_mesh(tetr_mesh_unscaled, "tetra", scale_factor=scale_factor)
    tetr_mesh_scaled.write(tetr_xdmf_scaled)

    tria_mesh_unscaled = geometry.create_mesh(tet_msh, "triangle")
    tria_mesh_unscaled.write(tria_xdmf_unscaled)
    tria_mesh_scaled = geometry.scale_mesh(tria_mesh_unscaled, "triangle", scale_factor=scale_factor)
    tria_mesh_scaled.write(tria_xdmf_scaled)
