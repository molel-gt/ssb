import os

import alphashape
import gmsh
import h5py
import matplotlib.pyplot as plt
import meshio
import numpy as np
import shapely
import warnings

from concavehull import concavehull
from descartes import PolygonPatch
from dolfinx import cpp, fem, io, mesh, nls, plot
from mpi4py import MPI
from petsc4py import PETSc

import commons, geometry, grapher, utils
warnings.simplefilter('ignore')


markers = commons.SurfaceMarkers()
cell_types = commons.CellTypes()


def mesh_surface(coords, xmax=470, ymax=470):
    points = {}
    new_points = {}
    count = 0
    for row in coords:
        points[(row[0], row[1])] = count
        count += 1
    points_set = set(points.keys())
    no_full_neighbors = set()
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
    img = np.asarray(plt.imread('data/current_constriction/test11.tif')[:, :, 0], dtype=np.uint8)
    img2 = img.copy()
    img2[0:5, :] = 0
    img2[-5:, :] = 0
    img2[:, 0:5] = 0
    img2[:, -5:,] = 0
    coords = np.asarray(np.argwhere(img2 == 1), dtype=np.int32)
    xmax = 470
    ymax = 470
    all_points = {}
    corner_points = [
        (0, 0, 0),
        (xmax, 0, 0),
        (xmax, ymax, 0),
        (0, ymax, 0)
    ]
    other_points = []
    count = 0
    for row in coords:
        all_points[(row[0], row[1])] = count
        count += 1
    gmsh_points = []
    points_view = {int(v): (int(k[0]), int(k[1])) for k, v in all_points.items()}
    graph = grapher.PixelGraph(points=points_view)
    graph.build_graph()
    graph.get_graph_pieces()
    graph.n_pieces

    Lx = 470
    Ly = 470
    Lz = 2
    gmsh.initialize()
    gmsh.model.add('area')
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
        points0.append(
            gmsh.model.occ.addPoint(*z0_points[i], meshSize=1)
        )
    for i in range(4):
        points1.append(
            gmsh.model.occ.addPoint(*zL_points[i], meshSize=1)
        )
    gmsh.model.occ.synchronize()
    for i in range(-1, 3):
        lines.append(
            gmsh.model.occ.addLine(points0[i], points0[i + 1])
        )
    for i in range(-1, 3):
        lines.append(
            gmsh.model.occ.addLine(points1[i], points1[i + 1])
        )
    # 1 --> 5
    lines.append(
        gmsh.model.occ.addLine(points0[1], points1[1])
    )
    # 2 --> 6
    lines.append(
        gmsh.model.occ.addLine(points0[2], points1[2])
    )
    # 3 --> 7
    lines.append(
        gmsh.model.occ.addLine(points0[3], points1[3])
    )
    # 0 --> 4
    lines.append(
        gmsh.model.occ.addLine(points0[0], points1[0])
    )
    gmsh.model.occ.synchronize()

    loops = []
    # xy sides
    loops.append(
        gmsh.model.occ.addCurveLoop(lines[:4])
    )
    loops.append(
        gmsh.model.occ.addCurveLoop(lines[4:8])
    )
    # xz sides
    loops.append(
        gmsh.model.occ.addCurveLoop([lines[1]] + [lines[8]] + [lines[5]] + [lines[11]])
    )
    loops.append(
        gmsh.model.occ.addCurveLoop([lines[3]] + [lines[9]] + [lines[7]] + [lines[10]])
    )
    # yz sides
    loops.append(
        gmsh.model.occ.addCurveLoop([lines[2]] + [lines[8]] + [lines[6]] + [lines[9]])
    )
    loops.append(
        gmsh.model.occ.addCurveLoop([lines[0]] + [lines[11]] + [lines[4]] + [lines[10]])
    )
    gmsh.model.occ.synchronize()

    side_loops = []
    insulated = []
    right = []
    left = []

    for p in graph.pieces:
        if len(p) < 50:
            continue
        arr = []
        for c in p:
            arr.append(points_view[c])
        alpha_shape = alphashape.alphashape(arr, 0.25)
        exterior = alpha_shape.exterior
        hull = []
        for c in exterior.coords:
            hull.append((c[0], c[1]))
        hull_arr = np.asarray(hull)
        hull_points = []
        for pp in hull[:-1]:
            hull_points.append(
                gmsh.model.occ.addPoint(int(pp[0]), int(pp[1]), 0)
            )
        gmsh.model.occ.synchronize()
        hull_lines = []
        for i in range(-1, len(hull_points) - 1):
            hull_lines.append(
                gmsh.model.occ.addLine(hull_points[i], hull_points[i + 1])
            )
        gmsh.model.occ.synchronize()
        hull_loop = gmsh.model.occ.addCurveLoop(hull_lines)
        side_loops.append(hull_loop)
    gmsh.model.occ.synchronize()

    insulated = []
    insulated += [gmsh.model.occ.addPlaneSurface((loops[0], *side_loops))]
    left = [gmsh.model.occ.addPlaneSurface((vv,)) for vv in side_loops]
    righ = [gmsh.model.occ.addPlaneSurface((loops[1],))]
    insulated += [gmsh.model.occ.addPlaneSurface((vv,)) for vv in loops[2:]]
    gmsh.model.occ.synchronize()
    lefttag = gmsh.model.addPhysicalGroup(2, left, markers.left_cc)
    righttag = gmsh.model.addPhysicalGroup(2, right, markers.right_cc)
    insulatedtag = gmsh.model.addPhysicalGroup(2, insulated, markers.insulated)
    gmsh.model.occ.synchronize()
    # extr = gmsh.model.occ.extrude(gmsh.model.getEntities(2), 0, 0, 25)
    # gmsh.model.occ.synchronize()
    # print(extr)
    print(gmsh.model.occ.getEntities(2), gmsh.model.getEntities(3))
    sloop = gmsh.model.occ.addSurfaceLoop(tuple([vv[1] for vv in gmsh.model.occ.getEntities(2)]))
    volz = gmsh.model.occ.addVolume((sloop,))
    gmsh.model.occ.synchronize()
    vols = gmsh.model.occ.getEntities(3)
    print(vols)
    volz = [vv[1] for vv in vols]
    gmsh.model.addPhysicalGroup(3, volz, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write("trial.msh")
    gmsh.finalize()

    outdir = f'mesh/study_2/470-470-{Lz}_000-000-000/'
    utils.make_dir_if_missing(outdir)
    scale_factor = [0.045, 0.045, 0.05]
    msh = meshio.read("trial.msh")

    tria_mesh_unscaled = geometry.create_mesh(msh, cell_types.triangle)
    tria_mesh_unscaled.write(f"{outdir}" + 'tria.xdmf')
    # tria_mesh_scaled = geometry.scale_mesh(tria_mesh_unscaled, cell_types.triangle, scale_factor=scale_factor)
    # tria_mesh_scaled.write(f"{outdir}" + 'tria.xdmf')

    tetr_mesh_unscaled = geometry.create_mesh(msh, cell_types.tetra)
    tetr_mesh_unscaled.write(f"{outdir}" + 'tetr.xdmf')
    # tetr_mesh_scaled = geometry.scale_mesh(tetr_mesh_unscaled, cell_types.tetra, scale_factor=scale_factor)
    # tetr_mesh_scaled.write(f"{outdir}" + 'tetr.xdmf')