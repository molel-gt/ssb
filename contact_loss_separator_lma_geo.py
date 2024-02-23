#!/usr/bin/env python

import json
import os
import subprocess
import sys
import timeit

import alphashape
import argparse
import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
import scipy
import warnings

import commons, configs, geometry, grapher, utils
warnings.simplefilter('ignore')


markers = commons.Markers()
cell_types = commons.CellTypes()
max_resolution = 2.5


def mesh_surface(coords, xmax=470, ymax=470):
    points = {}
    count = 0
    for row in coords:
        points[(row[0], row[1])] = count
        count += 1
    points_set = set(points.keys())

    triangles = []
    for (x0, y0) in points_set:
        p0 = points[(x0, y0)]
        neighbors = [
            (int(x0 + 1), y0),
            (int(x0 + 1), int(y0 + 1)),
            (x0, int(y0 + 1))
        ]
        neighbor_points = [p0]
        for p in neighbors:
            v = points.get(p)
            neighbor_points.append(v)

        midpoint = (x0 + 0.5, y0 + 0.5)
        if midpoint[0] > xmax or midpoint[1] > ymax:
            continue
        points[midpoint] = count
        p2 = count
        count += 1
        for i in range(4):
            p0 = neighbor_points[i]
            if i == 3:
                p1 = neighbor_points[0]
            else:
                p1 = neighbor_points[i + 1]
            if not p0 is None and not p1 is None:
                triangles.append(
                    (p0, p1, p2)
                )

    return triangles, points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimates Effective Conductivity.')
    parser.add_argument('--img_id', help='contact area image index', required=True, type=int)
    parser.add_argument("--dimensions", help="integer representation of Lx-Ly-Lz of the grid", required=True)
    parser.add_argument('--resolution', help=f'resolution with max resolution of 10x resolution', nargs='?', const=1, default=1, type=float)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', type=str, required=True)
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="contact_loss_lma")
    args = parser.parse_args()
    start_time = timeit.default_timer()
    Lx, Ly, Lz = [int(v) for v in args.dimensions.split("-")]
    scaling = configs.get_configs()[args.scaling]
    scale_x = float(scaling['x'])
    scale_y = float(scaling['y'])
    scale_z = float(scaling['z'])
    outdir = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], args.name_of_study, args.dimensions, str(args.img_id), str(args.resolution))
    utils.make_dir_if_missing(outdir)
    mshpath = os.path.join(f"{outdir}", "trial.msh")
    geometry_metafile = os.path.join(outdir, "geometry.json")
    corner_points = [
        (0, 0, 0),
        (Lx, 0, 0),
        (Lx, Ly, 0),
        (0, Ly, 0)
    ]

    min_resolution = (1/5) * args.resolution
    min_dist = 5e-5 * Lz

    gmsh.initialize()
    gmsh.model.add('area')
    gmsh.option.setNumber('Mesh.MeshSizeMin', min_resolution)
    gmsh.option.setNumber('Mesh.MeshSizeMax', args.resolution)
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    z0_points = [
        (0, 0, 0),
        (Lx, 0, 0),
        (Lx, Ly, 0),
        (0, Ly, 0),
    ]
    zL_points = [
        (0, 0, Lz),
        (Lx, 0, Lz),
        (Lx, Ly, Lz),
        (0, Ly, Lz),
    ]
    points0 = []
    points1 = []
    lines = []

    for i in range(4):
        idx = gmsh.model.occ.addPoint(*z0_points[i])
        points0.append(idx)
    for i in range(4):
        idx = gmsh.model.occ.addPoint(*zL_points[i])
        points1.append(
            idx
        )
    gmsh.model.occ.synchronize()
    for i in range(-1, 3):
        idx = gmsh.model.occ.addLine(points0[i], points0[i + 1])
        lines.append(
            idx
        )

    for i in range(-1, 3):
        idx = gmsh.model.occ.addLine(points1[i], points1[i + 1])
        lines.append(
            idx
        )

    # 1 --> 5
    idx = gmsh.model.occ.addLine(points0[1], points1[1])
    lines.append(
        idx
    )

    # 2 --> 6
    idx = gmsh.model.occ.addLine(points0[2], points1[2])
    lines.append(
        idx
    )

    # 3 --> 7
    idx = gmsh.model.occ.addLine(points0[3], points1[3])
    lines.append(
        idx
    )

    # 0 --> 4
    idx = gmsh.model.occ.addLine(points0[0], points1[0])
    lines.append(
        idx
    )

    gmsh.model.occ.synchronize()

    loops = []
    # xy sides
    idx = gmsh.model.occ.addCurveLoop(lines[:4])
    loops.append(
        idx
    )

    idx = gmsh.model.occ.addCurveLoop(lines[4:8])
    loops.append(
        idx
    )

    # xz sides
    idx = gmsh.model.occ.addCurveLoop([lines[1]] + [lines[8]] + [lines[5]] + [lines[11]])
    loops.append(
        idx
    )

    idx = gmsh.model.occ.addCurveLoop([lines[3]] + [lines[9]] + [lines[7]] + [lines[10]])
    loops.append(
        idx
    )

    # yz sides
    idx = gmsh.model.occ.addCurveLoop([lines[2]] + [lines[8]] + [lines[6]] + [lines[9]])
    loops.append(
        idx
    )

    idx = gmsh.model.occ.addCurveLoop([lines[0]] + [lines[11]] + [lines[4]] + [lines[10]])
    loops.append(
        idx
    )

    gmsh.model.occ.synchronize()

    side_loops = []
    insulated = []
    right = []
    left_active = []
    left = []
    process_count = 0
    img = np.asarray(plt.imread(f'data/current_constriction/test{str(int(args.img_id))}.tif')[:, :, 0], dtype=np.uint8)
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
            idx = gmsh.model.occ.addPoint(int(pp[0]), int(pp[1]), 0)
            hull_points.append(
                idx
            )
        gmsh.model.occ.synchronize()
        hull_lines = []
        for i in range(-1, len(hull_points) - 1):
            idx = gmsh.model.occ.addLine(hull_points[i], hull_points[i + 1])
            hull_lines.append(
                idx
            )

        gmsh.model.occ.synchronize()
        idx = gmsh.model.occ.addCurveLoop(hull_lines)
        side_loops.append(idx)
        idx2 = gmsh.model.occ.addPlaneSurface((idx, ))
        left.append(idx2)
        gmsh.model.occ.synchronize()

    right = [gmsh.model.occ.addPlaneSurface((loops[1], ))]
    gmsh.model.occ.synchronize()

    for vv in loops[2:]:
        idx = gmsh.model.occ.addPlaneSurface((vv, ))
        insulated.append(
            idx
        )
        gmsh.model.occ.synchronize()

    if len(np.unique(img)) == 1 and np.isclose(np.unique(img)[0], 1):
        insulated += [gmsh.model.occ.addPlaneSurface((loops[0], ))]
    else:
        insulated += [gmsh.model.occ.addPlaneSurface((loops[0], *side_loops))]

    gmsh.model.occ.healShapes()
    gmsh.model.occ.synchronize()
    print("Generating surface tags..")
    if len(np.unique(img)) == 1 and np.isclose(np.unique(img)[0], 1):
        lefttag = gmsh.model.addPhysicalGroup(2, [6], markers.left)
        righttag = gmsh.model.addPhysicalGroup(2, [1], markers.right)
        insulatedtag = gmsh.model.addPhysicalGroup(2, [2, 3, 4, 5], markers.insulated)
        surfaces = list(range(1, 7))
    else:
        left_surfs = [vv[1] for vv in gmsh.model.occ.getEntities(2) if vv[1] >= 7]
        lefttag = gmsh.model.addPhysicalGroup(2, left_surfs, markers.left)
        righttag = gmsh.model.addPhysicalGroup(2, [1], markers.right)
        insulatedtag = gmsh.model.addPhysicalGroup(2, [2, 3, 4, 5, 6], markers.insulated)
        surfaces = tuple(left + insulated + right)

    gmsh.model.occ.synchronize()
    sloop = gmsh.model.occ.addSurfaceLoop(surfaces)
    gmsh.model.occ.synchronize()
    physvol = gmsh.model.addPhysicalGroup(3, [1], 1)
    gmsh.model.occ.synchronize()

    # refinement
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "FacesList", [insulatedtag, lefttag, righttag])

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", 0.5 * args.resolution)
    gmsh.model.mesh.field.setNumber(2, "LcMax", args.resolution)
    gmsh.model.mesh.field.setNumber(2, "DistMin", min_dist)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 1)

    gmsh.model.mesh.field.add("Max", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)
    print("Generating mesh..")
    gmsh.model.mesh.generate(3)
    gmsh.write(f"{mshpath}")
    gmsh.finalize()
    # write to file
    scale_factor = [scale_x, scale_y, scale_z]
    msh = meshio.read(f"{mshpath}")

    tria_mesh_unscaled = geometry.create_mesh(msh, cell_types.triangle)
    tria_mesh_scaled = geometry.scale_mesh(tria_mesh_unscaled, cell_types.triangle, scale_factor=scale_factor)
    tria_mesh_scaled.write(os.path.join(outdir, 'tria.xdmf'))
    tetr_mesh_unscaled = geometry.create_mesh(msh, cell_types.tetra)
    tetr_mesh_scaled = geometry.scale_mesh(tetr_mesh_unscaled, cell_types.tetra, scale_factor=scale_factor)
    tetr_mesh_scaled.write(os.path.join(outdir, 'tetr.xdmf'))
    print("Cleaning up.")
    # clean up
    os.remove(mshpath)
    geometry_metadata = {
        "max_resolution": args.resolution,
        "dimensions": args.dimensions,
        "scaling": args.scaling,
    }
    with open(geometry_metafile, "w", encoding='utf-8') as f:
        json.dump(geometry_metadata, f, ensure_ascii=False, indent=4)
    print(f"Time elapsed                                    : {int(timeit.default_timer() - start_time):3.5f}s")
