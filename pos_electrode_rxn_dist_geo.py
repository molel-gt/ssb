#!/usr/bin/env python3
# coding: utf-8
import argparse
import os

import alphashape
import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
import pandas as pd
import scipy

import commons, configs, geometry, grapher, utils

markers = commons.Markers()
CELL_TYPES = commons.CellTypes()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Current Distribution')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="reaction_distribution")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    parser.add_argument("--resolution", help="maximum resolution", nargs='?', const=1, default=1, type=float)
    parser.add_argument("--L_SEP", help="integer representation of separator thickness", nargs='?', const=1, default=25, type=int)
    parser.add_argument("--R_PARTICLE", help="integer representation of AM particle radius", nargs='?', const=1, default=6, type=int)
    parser.add_argument("--am_vol_frac", help="volume fraction of active material phase", nargs='?', const=1, default=0.5, type=float)
    parser.add_argument("--void_vol_frac", help="volume fraction of void phase", nargs='?', const=1, default=0, type=float)
    parser.add_argument("--img_id", help="image id to identify contact map", nargs='?', const=1, default=11, type=int)
    parser.add_argument("--active_area_frac", help="active area fraction at neg electrode-SE interface", nargs='?', const=1, default=0.4, type=float)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='CONTACT_LOSS_SCALING', type=str)

    args = parser.parse_args()
    scaling = configs.get_configs()[args.scaling]
    scale_factor = [float(scaling[k]) for k in ['x', 'y', 'z']]
    LX, LY, LZ = [int(v) for v in args.dimensions.split("-")]
    outdir = f"output/{args.name_of_study}/{args.dimensions}/{args.L_SEP}/{args.active_area_frac}/{args.am_vol_frac}/{args.R_PARTICLE}/{args.img_id}/{args.resolution}"
    utils.make_dir_if_missing(outdir)
    msh_path = os.path.join(outdir, 'laminate.msh')
    tetr_path = os.path.join(outdir, 'tetr.xdmf')
    tria_path = os.path.join(outdir, 'tria.xdmf')
    Rp = args.R_PARTICLE
    R = 0.5 * LX

    Lcat = LZ - args.L_SEP
    df = 470 * pd.read_csv('data/laminate.csv')
    gmsh.initialize()
    gmsh.model.add('laminate')
    gmsh.option.setNumber('General.Verbosity', 1)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 1)

    # main section
    box_se = gmsh.model.occ.addCylinder(R, R, 0, 0, 0, args.L_SEP + Lcat - Rp, R)
    gmsh.model.occ.synchronize()
    box_am = gmsh.model.occ.addCylinder(R, R, args.L_SEP + Lcat - Rp, 0, 0, Rp, R)
    gmsh.model.occ.synchronize()
    cylinders = []
    spheres = []

    for idx in range(df.shape[0]):
        x, y, _ = df.loc[idx, :]
        if ((R - x) ** 2 + (R - y) ** 2) ** 0.5 >= 225:
            continue
        cyl = gmsh.model.occ.addCylinder(x, y, args.L_SEP, 0, 0, Lcat - Rp, Rp)
        gmsh.model.occ.synchronize()
        cylinders.append(cyl)
        gmsh.model.occ.synchronize()

    # 3D mesh
    se_phase = gmsh.model.occ.cut([(3, box_se)], [(3, c) for c in cylinders], removeTool=False)
    gmsh.model.occ.synchronize()
    union = gmsh.model.occ.fuse([(3, box_am)], [(3, c) for c in cylinders])
    gmsh.model.occ.synchronize()

    vols = gmsh.model.occ.getEntities(3)
    se_vol = gmsh.model.addPhysicalGroup(3, [vols[0][1]], markers.electrolyte)
    gmsh.model.occ.synchronize()
    gmsh.model.setPhysicalName(3, se_vol, "electrolyte")
    gmsh.model.occ.synchronize()
    am_vol = gmsh.model.addPhysicalGroup(3, [vols[1][1]], markers.positive_am)
    gmsh.model.occ.synchronize()
    gmsh.model.setPhysicalName(3, am_vol, "positive_am")
    gmsh.model.occ.synchronize()
    right = []
    left = []
    insulated = []
    insulated_se = []
    insulated_am = []
    interface = []
    main_left = []

    for surf in gmsh.model.occ.getEntities(2):
        com = gmsh.model.occ.getCenterOfMass(surf[0], surf[1])
        if np.isclose(com[2], args.L_SEP + Lcat):
            right.append(surf[1])
        elif np.isclose(com[2], 0):
            if np.isclose(com[0], R) and np.isclose(com[1], R):
                left.append(surf[1])
        elif np.isclose(com[0], 235) and np.isclose(com[1], 235) and np.isclose(com[2], 0.5 * (LZ - Rp)):
                insulated_se.append(surf[1])
        elif np.isclose(com[0], 235, atol=1) and np.isclose(com[1], 235, atol=2) and np.isclose(com[2], args.L_SEP + Lcat - 0.5 * Rp, atol=0.1):
                insulated_am.append(surf[1])
        else:
            interface.append(surf[1])

    # generate active labeled surface
    left_active = []
    img = np.asarray(plt.imread(f'data/current_constriction/test{args.img_id}.tif')[:, :, 0], dtype=np.uint8)
    image = img.copy()
    image[0, :] = 0
    image[-1, :] = 0
    image[:, 0] = 0
    image[:, -1] = 0
    for i in range(470):
        for j in range(470):
            if ((i - 235) ** 2 + (j - 235) ** 2) ** 0.5 > 235:
                image[i, j] = 0
    gmsh_points = []
    boundary_pieces, count, all_points, points_view = geometry.get_phase_boundary_pieces(image)
    for hull in boundary_pieces:
        hull_arr = np.asarray(hull)
        hull_points = []
        for pp in hull[:-1]:
            idx = gmsh.model.occ.addPoint(int(pp[0]), int(pp[1]), 0)
            hull_points.append(
                idx
            )
        gmsh.model.occ.synchronize()
        gmsh_points += hull_points
        hull_lines = []
        for i in range(-1, len(hull_points) - 1):
            idx = gmsh.model.occ.addLine(hull_points[i], hull_points[i + 1])
            hull_lines.append(
                idx
            )

        gmsh.model.occ.synchronize()
        idx = gmsh.model.occ.addCurveLoop(hull_lines)
        gmsh.model.occ.synchronize()
        idx2 = gmsh.model.occ.addPlaneSurface((idx, ))
        left_active.append(idx2)
        gmsh.model.occ.synchronize()

    left_inactive = gmsh.model.occ.cut([(2, k) for k in left], [(2, kk) for kk in left_active], removeTool=False, removeObject=False)
    print(left_inactive[0])
    gmsh.model.occ.synchronize()
    left_inactive = [v[1] for v in left_inactive[0]]
    left_tag = gmsh.model.addPhysicalGroup(2, left_active, markers.left)
    gmsh.model.setPhysicalName(2, left_tag, "left")
    gmsh.model.occ.synchronize()
    right_tag = gmsh.model.addPhysicalGroup(2, right, markers.right)
    gmsh.model.setPhysicalName(2, right_tag, "right")
    electrolyte_v_positive_am_tag = gmsh.model.addPhysicalGroup(2, interface, markers.electrolyte_v_positive_am)
    gmsh.model.setPhysicalName(2, electrolyte_v_positive_am_tag, "electrolyte_positive_am_interface")
    gmsh.model.occ.synchronize()
    insulated_am_tag = gmsh.model.addPhysicalGroup(2, insulated_am, markers.insulated_positive_am)
    gmsh.model.setPhysicalName(2, insulated_am_tag, "insulated_am")
    insulated_se += left_inactive
    insulated = insulated_se + insulated_am
    insulated_se_tag = gmsh.model.addPhysicalGroup(2, insulated_se, markers.insulated_electrolyte)
    gmsh.model.setPhysicalName(2, insulated_se_tag, "insulated_electrolyte")
    insulated_tag = gmsh.model.addPhysicalGroup(2, insulated, markers.insulated)
    gmsh.model.setPhysicalName(2, insulated_tag, "insulated")

    # refinement
    print("Performing local refinement..")
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "FacesList", [left_tag, insulated_tag, electrolyte_v_positive_am_tag, right_tag])

    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "LcMin", args.resolution)
    gmsh.model.mesh.field.setNumber(threshold, "LcMax", 10 * args.resolution)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.5)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", 5)

    max_f = gmsh.model.mesh.field.add("Max")
    gmsh.model.mesh.field.setNumbers(max_f, "FieldsList", [threshold])
    gmsh.model.mesh.field.setAsBackgroundMesh(max_f)
    print("Generating mesh..")
    gmsh.model.mesh.generate(3)
    gmsh.write(msh_path)
    gmsh.finalize()

    msh = meshio.read(msh_path)
    tetr_unscaled = geometry.create_mesh(msh, CELL_TYPES.tetra)
    tetr_unscaled.write(tetr_path)
    tetr_scaled = geometry.scale_mesh(tetr_unscaled, CELL_TYPES.tetra, scale_factor=scale_factor)
    tetr_scaled.write(tetr_path)
    tria_unscaled = geometry.create_mesh(msh, CELL_TYPES.triangle)
    tria_unscaled.write(tria_path)
    tria_scaled = geometry.scale_mesh(tria_unscaled, CELL_TYPES.triangle, scale_factor=scale_factor)
    tria_scaled.write(tria_path)
