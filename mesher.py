#!/usr/bin/env python3
import argparse
import math
import os
import sys

import gmsh
import numpy as np
import pymeshlab
import trimesh
import timeit

import commons, utils


PHASE_VALUES = {"voids": 0, "sse": 1, "cam": 2}

def group_surfaces_adjacencies(adj):
    '''
    Function to goup the surfaces adjacencies.
    Reference: https://stackoverflow.com/a/4842897
    '''
    l = adj
    out = []
    while len(l)>0:
        first, *rest = l
        first = set(first)

        lf = -1
        while len(first)>lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)     
            rest = rest2

        out.append(list(first))
        l = rest
    return out


def create_box_surface_loop(Lx, Ly, Lz, L_sep, origin):
    x0, y0, z0 = origin#[-1e-6, -1e-6, -1e-6]
    coords = [
        (x0, y0, z0),
        (L_sep + Lx, y0, z0),
        (L_sep + Lx, Ly, z0),
        (x0, Ly, z0),
        (x0, y0, Lz),
        (L_sep + Lx, y0, Lz),
        (L_sep + Lx, Ly, Lz),
        (x0, Ly, Lz),
    ]
   
    points = [gmsh.model.geo.addPoint(*p) for p in coords]
    lines = [gmsh.model.geo.addLine(points[i], points[i+1]) for i in range(4-1)]
    lines.append(gmsh.model.geo.addLine(points[3], points[0])) # line 4
    lines.extend([gmsh.model.geo.addLine(points[i], points[i+1]) for i in range(4, 7)])
    lines.append(gmsh.model.geo.addLine(points[7], points[4])) # line 7
    lines.append(gmsh.model.geo.addLine(points[3], points[7])) # line 8
    lines.append(gmsh.model.geo.addLine(points[4], points[0])) # line 9
    lines.append(gmsh.model.geo.addLine(points[1], points[5])) # line 10
    lines.append(gmsh.model.geo.addLine(points[2], points[6])) # line 11
    loops = []
    # gmsh.model.geo.synchronize()
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [0, 1, 2, 3]], reorient=True))
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [4, 5, 6, 7]], reorient=True))
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [8, 7, 9, 3]], reorient=True))
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [9, 0, 10, 4]], reorient=True))
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [10, 5, 11, 1]], reorient=True))
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [11, 6, 8, 2]], reorient=True))
    # gmsh.model.geo.synchronize()
    surfs = [gmsh.model.geo.addPlaneSurface([loop]) for loop in loops]
    # gmsh.model.geo.synchronize()
    surf_loop = gmsh.model.geo.addSurfaceLoop(surfs)

    return surf_loop


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='secondary current distribution')
    parser.add_argument('--size', help='Lx-Ly-Lz', required=True, type=str)
    parser.add_argument("--origin", help="where to extract data", nargs='?', const=1, default='0-0-0', type=str)
    parser.add_argument('--scale', help='sx-sy-sz', required=True, type=str)
    parser.add_argument("--L_sep", help="separator thickness [m]", nargs='?', const=1, default=15e-6, type=float)
    parser.add_argument("--resolution", help="dimensionless resolution", nargs='?', const=1, default=0.05, type=float)
    args = parser.parse_args()
    start_time = timeit.default_timer()
    scaling = [float(v) for v in args.scale.split(",")]
    markers = commons.Markers()
    workdir = os.path.join(f"output/segmentation/{args.size}/{args.origin}")
    utils.make_dir_if_missing(workdir)
    x0, y0, z0 = [int(val) for val in args.origin.split("-")]
    L_SEP = args.L_sep
    nx, ny, nz = [int(s) for s in args.size.split("-")]
    LX = nx #- 1
    LY = ny - 1
    LZ = nz - 1
    L_c = LX * scaling[0] + L_SEP
    Lx = LX / L_c
    Ly = LY * scaling[1] / L_c
    Lz = LZ * scaling[2] / L_c
    L_sep = L_SEP / L_c
    padding = 1e-6 / L_c
    non_dim_scale = [scaling[0]/L_c, scaling[1]/L_c, scaling[2]/L_c]
    gmsh.initialize()
    gmsh.model.add("fib_sem")
    gmsh.onelab.set("""
        [
          {
            "type":"number",
            "name":"Parameters/Angle for surface detection",
            "values":[180],
            "min":20,
            "max":120,
            "step":1
          },
          {
            "type":"number",
            "name":"Parameters/Create surfaces guaranteed to be parametrizable",
            "values":[1],
            "choices":[0, 1]
          },
          {
            "type":"number",
            "name":"Parameters/Apply funny mesh size field?",
            "values":[0],
            "choices":[0, 1]
          }
          ]
    """)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1)
    # gmsh.model.mesh.setOrder(1)
    # gmsh.option.setNumber('Geometry.Tolerance', 1e-7)
    # gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.01)
    # gmsh.option.setNumber('Mesh.Optimize', 1)
    # gmsh.option.setNumber('Mesh.Algorithm', 5)
    gmsh.option.setNumber("General.NumThreads", 8)
    # gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.002)
    #gmsh.option.setNumber("Mesh.ScalingFactor", 0.05e-6/L_c)
    # gmsh.option.setNumber("General.Verbosity", 1)
    angle = gmsh.onelab.getNumber('Parameters/Angle for surface detection')[0]
    forceParametrizablePatches = gmsh.onelab.getNumber(
        'Parameters/Create surfaces guaranteed to be parametrizable')[0]
    curveAngle = 180
    threshold = 0
    threshold_surf = 0
    phase_volumes = {}
    phase_surfaces = {}
    # for phase in ["cam", "sse"]:
    #     gmsh.merge(f"{phase}.vtk")
    #     gmsh.model.mesh.createTopology()
    #     gmsh.model.mesh.classifySurfaces(angle * math.pi/180., True, True, curveAngle * math.pi/180.)
    #     gmsh.model.mesh.createGeometry()
    #     vols = [v[1] for v in gmsh.model.getEntities(3) if v[1] > threshold]
    #     phase_volumes[phase] = vols
    #     threshold = max(vols)
    #     gmsh.model.geo.synchronize()
    #     gmsh.model.geo.removeAllDuplicates()
    #     gmsh.model.geo.synchronize()
    gmsh.merge(os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/composite-electrode/mesh.mesh"))
    # gmsh.model.mesh.createTopology()
    # gmsh.model.mesh.classifySurfaces(angle * math.pi/180., False, False, curveAngle * math.pi/180.)
    # gmsh.model.mesh.createGeometry()
    gmsh.model.geo.synchronize()
    vols = gmsh.model.getEntities(3)
    surfs = gmsh.model.getEntities(2)
    print(vols)
    print(surfs)
    gmsh.model.addPhysicalGroup(3, [markers.positive_am], markers.positive_am, "CAM")
    gmsh.model.addPhysicalGroup(3, [markers.electrolyte], markers.electrolyte, "SSE")
    gmsh.model.geo.synchronize()
    # surfs = [s[1] for s in gmsh.model.getEntities(2)]
    # cam_boundary = [s[1] for s in gmsh.model.getBoundary([(3, v) for v in phase_volumes["cam"]], combined=False) if s[0] == 2]
    # sse_boundary = [s[1] for s in gmsh.model.getBoundary([(3, v) for v in phase_volumes["sse"]], combined=False) if s[0] == 2]
    # interface = set(tuple(cam_boundary)).union(set(tuple(sse_boundary)))
    # tol = 1e-4
    # left_surfs = [s[1] for s in gmsh.model.getEntitiesInBoundingBox(-tol, -tol, -tol, tol, 2, 2, dim=2) if s[0] == 2]
    # # print(left_surfs)
    # # x_vals = []
    # # # xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, phase_volumes["sse"][0])
    # # for v in phase_volumes["sse"]:
    # #     _xmin, _ymin, _zmin, _xmax, _ymax, _zmax = gmsh.model.getBoundingBox(3, v)
    # #     x_vals.extend([_xmin, _xmax])
    # #     # xmax = max([xmax, _xmax])
    # # for xval in x_vals:
    # #     if xval > 0.99:
    # #         print(xval)

    # # right_surfs = [s[1] for s in gmsh.model.getEntitiesInBoundingBox(xmax-tol, -tol, -tol, xmax+tol, 2, 2, dim=2) if s[0] == 2]
    # # print(right_surfs)
    # right_surfs = []
    # for surf in sse_boundary:
    #     _xmin, _ymin, _zmin, _xmax, _ymax, _zmax = gmsh.model.getBoundingBox(2, surf)
    #     if _xmin > 0.99 or _xmax > 0.99:
    #         right_surfs.append(surf)
    # print(right_surfs)
    # quit()
    gmsh.model.addPhysicalGroup(2, [markers.electrolyte_v_positive_am], markers.electrolyte_v_positive_am, "positive charge xfer")
    gmsh.model.addPhysicalGroup(2, [markers.left], markers.left, "left")
    gmsh.model.addPhysicalGroup(2, [markers.right], markers.right, "right")
    # gmsh.model.addPhysicalGroup(2, surfs, 0, "Surfaces")

    gmsh.write(f"mesh.geo_unrolled")
    gmsh.model.mesh.generate(3)
    gmsh.write(os.path.join(workdir, f"mesh.msh"))
    print(f"Time elapsed {timeit.default_timer()-start_time:,.0f}s")
