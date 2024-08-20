#!/usr/bin/env python3
import argparse
import json
import os
import timeit

import gmsh
import numpy as np

import commons, utils


markers = commons.Markers()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="reaction_distribution")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid',  nargs='?', const=1, default='40-40-75')
    parser.add_argument('--resolution', help=f'max resolution (units of microns)', nargs='?', const=1, default=1, type=np.float16)
    parser.add_argument("--refine", help="compute current distribution stats", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    if args.refine:
        workdir = os.path.join("output", args.name_of_study, args.dimensions, str(args.resolution))
    else:
        workdir = os.path.join("output", args.name_of_study, args.dimensions, "unrefined", str(args.resolution))
    utils.make_dir_if_missing(workdir)
    mshpath = os.path.join(workdir, "mesh.msh")
    geometry_metafile = os.path.join(workdir, "geometry.json")
    start_time = timeit.default_timer()
    resolution = args.resolution * 1e-6
    gmsh.initialize()
    gmsh.model.add('cell')
    if not args.refine:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", resolution)
    cyl = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 75e-6, 20e-6)
    circle = gmsh.model.occ.addCircle(0, 0, 65e-6, 10e-6)
    curved_surf = gmsh.model.occ.extrude([(1, circle)], 0, 0, -40e-6)
    final_circle = [c for c in curved_surf if c[0] == 1]
    curved_surfaces = [c for c in curved_surf if c[0] == 2]

    for c in final_circle:
        com = gmsh.model.occ.getCenterOfMass(*c)
        if np.isclose(com[2], 25e-6):
            final_disk = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([c[1]])])
    start_disk = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([circle])])
    copy = gmsh.model.occ.copy([(1, circle)])
    gmsh.model.occ.dilate(copy, 0, 0, 65e-6, 2, 2, 0)
    main_disk = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([copy[0][1]])])
    ring, _ = gmsh.model.occ.cut([(2, main_disk)], [(2, start_disk)])
    vols, _ = gmsh.model.occ.fragment([(3, cyl)], [(2, final_disk)] + curved_surfaces + ring)
    volumes = [v[1] for v in vols if v[0] == 3]
    gmsh.model.occ.synchronize()
    for c in volumes:
        com = gmsh.model.occ.getCenterOfMass(3, c)
        if np.isclose(gmsh.model.getBoundingBox(3, c)[-1], 65e-6, atol=1e-6):
            gmsh.model.addPhysicalGroup(3, [c], markers.electrolyte, "electrolyte")
        else:
            gmsh.model.addPhysicalGroup(3, [c], markers.positive_am, "positive am")
    left = []
    right = []
    insulated_se = []
    insulated_am = []
    interface = []
    for surf in gmsh.model.occ.getEntities(2):
        com = gmsh.model.occ.getCenterOfMass(*surf)
        if np.isclose(com[2], 0):
            left.append(surf[1])
        elif np.isclose(com[2], 75e-6, atol=1e-6):
            right.append(surf[1])
        elif np.isclose(com[2], 0.5 * (65 + 75) * 1e-6, atol=1e-6):
            insulated_am.append(surf[1])
        elif np.isclose(com[2], 0.5 * 65e-6, atol=1e-6):
            insulated_se.append(surf[1])
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
        gmsh.model.mesh.field.setNumbers(1, "FacesList", left + interface + right + insulated_se + insulated_am)

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", resolution / 5)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", resolution)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 1e-6)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 2e-6)

        gmsh.model.mesh.field.add("Max", 5)
        gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(5)
        gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(mshpath)
    gmsh.finalize()
    work_time = int(timeit.default_timer() - start_time)
    metadata = {
        "resolution": resolution,
        "adaptive refine": args.refine,
        "Time elapsed (s)": work_time,
    }
    with open(geometry_metafile, "w", encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    print(f"Wrote {mshpath}")
    print(f"Time elapsed (s): {work_time}")
