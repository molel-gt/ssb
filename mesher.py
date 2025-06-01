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

SCALING = [1, 1, 1]#[0.0858e-6, 0.0858e-6, 0.05e-6]
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
    parser.add_argument("--phase", help="particulate phase", nargs='?', const=1, default='cam', type=str)
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
    LX = nx - 1
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
    # gmsh.option.setNumber("Mesh.Algorithm", 6)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1)
    gmsh.model.mesh.setOrder(1)
    gmsh.option.setNumber('Geometry.Tolerance', 1e-8)
    # gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.01)
    # gmsh.option.setNumber('Mesh.Optimize', 1)
    gmsh.option.setNumber('Mesh.Algorithm', 5)
    gmsh.option.setNumber("General.NumThreads", 8)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
    gmsh.option.setNumber("Mesh.MeshSizeMax", args.resolution)
    #gmsh.option.setNumber("Mesh.ScalingFactor", 0.05e-6/L_c)
    # gmsh.option.setNumber("General.Verbosity", 1)
    angle = gmsh.onelab.getNumber('Parameters/Angle for surface detection')[0]
    forceParametrizablePatches = gmsh.onelab.getNumber(
        'Parameters/Create surfaces guaranteed to be parametrizable')[0]
    curveAngle = 180
    threshold = 0
    phase_volumes = {}
    for phase in ["voids", "sse", "cam"]:
        print(threshold)
        gmsh.merge(f"output/segmentation/{args.size}/{args.origin}/{phase}.1.vtk")
        vols = [v[1] for v in gmsh.model.getEntities(3) if v[1] > threshold]
        phase_volumes[phase] = vols
        threshold = max(vols)
    # gmsh.model.geo.synchronize()

    # gmsh.model.geo.removeAllDuplicates()
    # gmsh.model.geo.synchronize()

    gmsh.model.mesh.createTopology()
    gmsh.model.mesh.classifySurfaces(angle * math.pi/180., True, forceParametrizablePatches, curveAngle * math.pi/180.)
    #gmsh.model.mesh.createGeometry()
    gmsh.model.geo.synchronize()
    gmsh.model.geo.removeAllDuplicates()
    gmsh.model.geo.synchronize()
    all_vols = [v[1] for v in gmsh.model.getEntities(3)]
    void_vols = [v for v in phase_volumes["voids"] if v in all_vols]
    sse_vols = [v for v in phase_volumes["sse"] if v in all_vols]
    cam_vols = [v for v in phase_volumes["cam"] if v in all_vols]
    print(np.min(all_vols), np.max(all_vols), len(all_vols))
    print(void_vols)
    print(sse_vols)
    print(cam_vols)
    gmsh.model.addPhysicalGroup(3, void_vols, markers.void, "VOIDS")
    gmsh.model.addPhysicalGroup(3, cam_vols, markers.positive_am, "CAM")
    gmsh.model.addPhysicalGroup(3, sse_vols, markers.electrolyte, "SSE")
    gmsh.model.geo.synchronize()
    phys_vols = gmsh.model.getPhysicalGroups(3)

    left_surfs = []
    right_surfs = []
    interface_surfs = []
    insulated_am = []
    insulated_se = []
    surfs = gmsh.model.getEntities(2)
    tol = 1e-8
    for surf in surfs:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.get_bounding_box(*surf)
        if np.isclose(xmin, 0, atol=tol) and np.isclose(xmax, 0, atol=tol):
            right_surfs.append(surf[1])
        elif np.isclose(xmin, 1.0, atol=tol) and np.isclose(xmax, 1.0, atol=tol):
            left_surfs.append(surf[1])
        elif np.isclose(ymin, ymax, atol=tol) and (np.isclose(ymin, 0, atol=tol) or np.isclose(ymax, Ly, atol=tol)):
            insulated_am.append(surf[1])
        elif np.isclose(zmin, zmax, atol=tol) and (np.isclose(zmin, 0, atol=tol) or np.isclose(zmax, Lz, atol=tol)):
            insulated_am.append(surf[1])
        else:
            interface_surfs.append(surf[1])
    gmsh.model.addPhysicalGroup(2, left_surfs, markers.left, "Left")
    gmsh.model.addPhysicalGroup(2, right_surfs, markers.right, "Right")
    gmsh.model.addPhysicalGroup(2, interface_surfs, markers.electrolyte_v_positive_am, "SE/AM")
    gmsh.model.geo.synchronize()

    gmsh.write("mesh.geo_unrolled")
    gmsh.model.mesh.generate(3)
    gmsh.write(os.path.join(workdir, "mesh.msh"))
    print(f"Time elapsed {timeit.default_timer()-start_time:,.0f}s")
