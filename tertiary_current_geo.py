#!/usr/bin/env python3
import argparse
import os

import gmsh
import numpy as np

import commons, utils

markers = commons.Markers()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument("-n", "--name_of_study", help="name_of_study", nargs='?', const=1, default="reaction_distribution")
    parser.add_argument("-d", '--dimensions', help='integer representation of Lx-Ly-Lz of the grid',  nargs='?', const=1, default='20-20-80')
    parser.add_argument("-r", '--resolution', help=f'max resolution (units of microns)', nargs='?', const=1, default=1, type=float)
    parser.add_argument("-f", "--refine", help="compute current distribution stats", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-hexahedron", "--hexahedron", help="compute current distribution stats", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-format", "--format", help="Mesh format", default="msh", nargs='?', const=1)
    args = parser.parse_args()

    workdir = os.path.join("output", args.name_of_study, args.dimensions, str(args.resolution))
    if args.refine:
        workdir = os.path.join("output", args.name_of_study, args.dimensions, str(args.resolution), "refined")

    L_CELL = 80
    L_SEP = 25
    L_slab_am = 5
    LY = 20
    LX = 20

    utils.make_dir_if_missing(workdir)
    mshpath = os.path.join(workdir, f"mesh.{args.format}")
    geometry_metafile = os.path.join(workdir, "geometry.json")
    gmsh.initialize()
    gmsh.model.add('ellipsoidals')
    gmsh.option.setNumber("Mesh.MeshSizeMax", args.resolution)
    # gmsh.option.setNumber('Geometry.ToleranceBoolean', 0.001)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumCirclePoints", 20)
    gmsh.option.setNumber('Mesh.MinimumElementsPerTwoPi', 15)
    # gmsh.option.setNumber('Mesh.Algorithm', 6)

    box_am = gmsh.model.occ.addBox(-0.5*LX/L_CELL, -0.5*LY/L_CELL, (L_CELL - L_slab_am)/L_CELL, LX/L_CELL, LY/L_CELL, L_slab_am/L_CELL)
    ellipsoids = []

    lxs = np.arange(-0.5*LX/L_CELL+2.5/80, 0.5*LX/L_CELL, 5/L_CELL)
    lys = np.arange(-0.5*LY/L_CELL+2.5/80, 0.5*LY/L_CELL, 5/L_CELL)

    z_pos = (L_CELL - L_slab_am)/L_CELL
    while z_pos > 0.3175:
        for x in lxs:
            for y in lys:
                sphere = gmsh.model.occ.addSphere(x, y, z_pos, 2/L_CELL)
                gmsh.model.occ.dilate([(3, sphere)], x, y, z_pos, 1, 1, 5/4)
                ellipsoids.append((3, sphere))
                gmsh.model.occ.synchronize()
        z_pos -= 4.0/L_CELL

    gmsh.model.occ.synchronize()

    ov, ovv = gmsh.model.occ.fuse([(3, box_am)], ellipsoids)
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    box_se = gmsh.model.occ.addBox(-0.5*LX/L_CELL, -0.5*LY/L_CELL, 0, LX/L_CELL, LY/L_CELL, 1)
    gmsh.model.occ.synchronize()
    res = gmsh.model.occ.cut([(3, box_se)], vols, removeTool=False)
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    gmsh.model.addPhysicalGroup(3, [vols[1][1]], markers.electrolyte, "electrolyte")
    gmsh.model.addPhysicalGroup(3, [vols[0][1]], markers.positive_am, "positive am")
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

    gmsh.model.addPhysicalGroup(2, left, markers.left, "left")
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
