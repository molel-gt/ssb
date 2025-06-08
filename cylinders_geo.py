#!/usr/bin/env python3
import argparse
import os

import gmsh
import matplotlib.pyplot as plt
import numpy as np
import alphashape

import commons, configs, geometry, grapher, utils

markers = commons.Markers()

area_frac_to_img_id = {
    "98.41": 6,
    "36.38": 11,
    "6.30": 16,
    "0.45": 22,
}


def build_active_contact_area_map(img, scale_x, scale_y, LX, LY, L_CELL):
    max_surf_id = max([s[1] for s in gmsh.model.occ.getEntities(2)])
    side_loops = []
    insulated = []
    right = []
    left_active = []
    left = []
    process_count = 0
    image = img.copy()
    image[0, :] = 0
    image[-1, :] = 0
    image[:, 0] = 0
    image[:, -1] = 0
    boundary_pieces, count, points, points_view = geometry.get_phase_boundary_pieces(image)
    for hull in boundary_pieces:
        hull_arr = np.asarray(hull)
        hull_points = []
        for pp in hull[:-1]:
            idx = gmsh.model.occ.addPoint(int(pp[0]) * scale_x - 0.5*LX/L_CELL, int(pp[1]) * scale_y - 0.5*LY/L_CELL, 0)
            hull_points.append(
                idx
            )
        hull_lines = []
        for i in range(-1, len(hull_points) - 1):
            idx = gmsh.model.occ.addLine(hull_points[i], hull_points[i + 1])
            hull_lines.append(
                idx
            )
        idx = gmsh.model.occ.addCurveLoop(hull_lines)
        side_loops.append(idx)
        idx2 = gmsh.model.occ.addPlaneSurface((idx, ))
        left.append(idx2)
    return left


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument("-n", "--name_of_study", help="name_of_study", nargs='?', const=1, default="cylinders")
    parser.add_argument("-d", '--dimensions', help='integer representation of Lx-Ly-Lz of the grid',  nargs='?', const=1, default='20-20-80')
    parser.add_argument("-r", '--resolution', help=f'max resolution (units of microns)', nargs='?', const=1, default=1, type=float)
    parser.add_argument("-f", "--refine", help="compute current distribution stats", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-hexahedron", "--hexahedron", help="compute current distribution stats", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-format", "--format", help="Mesh format", default="msh", nargs='?', const=1)
    parser.add_argument("-A", "--active_area_fraction", help="active area fraction", default=1, nargs='?', const=1, type=float)
    args = parser.parse_args()

    L_CELL = 80
    L_SEP = 25
    L_slab_am = 5
    LY = 20
    LX = 20
    scale_x = LX/L_CELL * 1.0/470.0
    scale_y = LY/L_CELL * 1.0/470.0
    img_id = area_frac_to_img_id.get(f"{args.active_area_fraction:.2f}")
    if not np.isclose(args.active_area_fraction, 1) and img_id is not None:
        img = np.asarray(plt.imread(f'data/current_constriction/test{str(int(img_id))}.tif')[:, :, 0], dtype=np.uint8)

    workdir = os.path.join("output", args.name_of_study, args.dimensions, str(args.resolution), f"{args.active_area_fraction:.2f}")
    if args.refine:
        workdir = os.path.join("output", args.name_of_study, args.dimensions, str(args.resolution), f"{args.active_area_fraction:.2f}", "refined")

    utils.make_dir_if_missing(workdir)
    mshpath = os.path.join(workdir, f"mesh.{args.format}")
    geometry_metafile = os.path.join(workdir, "geometry.json")
    gmsh.initialize()
    gmsh.model.add('ellipsoidals')
    gmsh.option.setNumber("Mesh.MeshSizeMax", args.resolution)
    gmsh.option.setNumber('Mesh.Optimize', 1)
    gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.85)
    # gmsh.option.setNumber("Mesh.MinimumCirclePoints", 20)
    # gmsh.option.setNumber('Mesh.MinimumElementsPerTwoPi', 10)
    # gmsh.option.setNumber('Mesh.Algorithm3D', 9)
    gmsh.option.setNumber("Mesh.ColorCarousel", 2)

    box_am = gmsh.model.occ.addBox(-0.5*LX/L_CELL, -0.5*LY/L_CELL, (L_CELL - L_slab_am)/L_CELL, LX/L_CELL, LY/L_CELL, L_slab_am/L_CELL)
    cylinders = []

    lxs = np.arange(-0.5*LX/L_CELL+2.5/80, 0.5*LX/L_CELL, 5/L_CELL)
    lys = np.arange(-0.5*LY/L_CELL+2.5/80, 0.5*LY/L_CELL, 5/L_CELL)

    z_pos = (L_CELL - L_slab_am)/L_CELL
    for x in lxs:
        for y in lys:
            sphere = gmsh.model.occ.addCylinder(x, y, L_SEP/L_CELL, 0, 0, 1 - (L_SEP + L_slab_am)/L_CELL, 2/L_CELL)
            cylinders.append((3, sphere))
            gmsh.model.occ.synchronize()
        # z_pos -= 4.0/L_CELL

    fused, fused2 = gmsh.model.occ.fuse(cylinders[:1], cylinders[1:])
    gmsh.model.occ.synchronize()
    tol = 0.1/L_CELL * 10
    vols = gmsh.model.getEntities(3)
    join = gmsh.model.occ.getEntitiesInBoundingBox(-0.5*LX/L_CELL - tol, -0.5*LY/L_CELL - tol, L_SEP/L_CELL - tol, LX/L_CELL, LY/L_CELL, 1-L_slab_am/L_CELL - tol)
    ov = gmsh.model.occ.fillet([v[1] for v in vols[1:]], [i[1] for i in join if i[0] == 1], [tol], removeVolume=True)
    gmsh.model.occ.synchronize()
    ov, ovv = gmsh.model.occ.fuse([(3, box_am)], ov)
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    box_se = gmsh.model.occ.addBox(-0.5*LX/L_CELL, -0.5*LY/L_CELL, 0, LX/L_CELL, LY/L_CELL, 1)
    gmsh.model.occ.synchronize()
    res = gmsh.model.occ.cut([(3, box_se)], vols, removeTool=False)
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    centers = []
    for v in vols:
        com = gmsh.model.occ.getCenterOfMass(*v)
        centers.append(com[2])
    if centers[0] > centers[1]:
        se_vols = [vols[1][1]]
        am_vols = [vols[0][1]]
    else:
        se_vols = [vols[0][1]]
        am_vols = [vols[1][1]]
    left_active = build_active_contact_area_map(img, scale_x, scale_y, LX, LY, L_CELL)
    ov, ovv = gmsh.model.occ.fragment([(3, se_vols[0])], [(2, s) for s in left_active], removeTool=False)
    surfs = [s[1] for s in ov if s[0] == 2]
    se_vols = surfs = [s[1] for s in ov if s[0] == 3]
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, se_vols, markers.electrolyte, "electrolyte")
    gmsh.model.addPhysicalGroup(3, am_vols, markers.positive_am, "positive am")
    left = []
    right = []
    insulated_am = []
    insulated_se = []
    interface = []
    main_xface = []
    pieces = []
    for surf in gmsh.model.getEntities(2):
        com = gmsh.model.occ.getCenterOfMass(*surf)
        area = gmsh.model.occ.getMass(*surf)
        if np.isclose(com[2], 0):
            left.append(surf[1])
            continue
        elif np.isclose(com[2], 1):
            right.append(surf[1])
            continue
        elif np.isclose(com[2], 1 - 0.5 * L_slab_am/L_CELL):
            insulated_am.append(surf[1])
            continue
        elif np.isclose(com[2], 0.5 * (L_CELL - L_slab_am)/L_CELL):
            insulated_se.append(surf[1])
            continue
        else:
            if np.isclose(area, (LX/L_CELL) ** 2):
                main_xface.append(surf)
            else:
                pieces.append(surf)
            if not np.isclose(com[2], 1 - L_slab_am/L_CELL, atol=1e-6):
                interface.append(surf[1])
            else:
                interface.append(surf[1])

    if img_id is not None:
        gmsh.model.addPhysicalGroup(2, left_active, markers.left, "left")
    else:
        gmsh.model.addPhysicalGroup(2, left, markers.left, "left")
    # gmsh.model.setColor([(2, s) for s in left], 255, 0, 0)
    gmsh.model.addPhysicalGroup(2, right, markers.right, "right")
    gmsh.model.addPhysicalGroup(2, insulated_am, markers.insulated_positive_am, "insulated_positive_am")
    gmsh.model.addPhysicalGroup(2, insulated_se, markers.insulated_electrolyte, "insulated_electrolyte")
    gmsh.model.addPhysicalGroup(2, interface, markers.electrolyte_v_positive_am, "electrolyte_v_positive_am")
    gmsh.model.occ.synchronize()
    if args.refine:
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "FacesList", interface)

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", args.resolution / 5)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", args.resolution)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.002)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.01)

        gmsh.model.mesh.field.add("Max", 5)
        gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(5)
        gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(mshpath)
    gmsh.finalize()
