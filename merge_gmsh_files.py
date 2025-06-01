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


if __name__ == "__main__":
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
    for phase in ["sse", "cam"]:
        gmsh.merge(f"output/segmentation/{args.size}/{args.origin}/{phase}.msh")
    gmsh.model.geo.synchronize()

    # left_surfs = []
    # right_surfs = []
    # interface_surfs = []
    # insulated_am = []
    # insulated_se = []
    # surfs = gmsh.model.getEntities(2)
    # tol = 1e-8
    # for surf in surfs:
    #     xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.get_bounding_box(*surf)
    #     if np.isclose(xmin, 0, atol=tol) and np.isclose(xmax, 0, atol=tol):
    #         right_surfs.append(surf[1])
    #     elif np.isclose(xmin, 1.0, atol=tol) and np.isclose(xmax, 1.0, atol=tol):
    #         left_surfs.append(surf[1])
    #     elif np.isclose(ymin, ymax, atol=tol) and (np.isclose(ymin, 0, atol=tol) or np.isclose(ymax, Ly, atol=tol)):
    #         insulated_am.append(surf[1])
    #     elif np.isclose(zmin, zmax, atol=tol) and (np.isclose(zmin, 0, atol=tol) or np.isclose(zmax, Lz, atol=tol)):
    #         insulated_am.append(surf[1])
    #     else:
    #         interface_surfs.append(surf[1])
    # gmsh.model.addPhysicalGroup(2, left_surfs, markers.left, "Left")
    # gmsh.model.addPhysicalGroup(2, right_surfs, markers.right, "Right")
    # gmsh.model.addPhysicalGroup(2, interface_surfs, markers.electrolyte_v_positive_am, "SE/AM")
    # gmsh.model.geo.synchronize()

    gmsh.write(f"mesh.geo_unrolled")
    gmsh.model.mesh.generate(3)
    gmsh.write(os.path.join(workdir, f"mesh.msh"))
    print(f"Time elapsed {timeit.default_timer()-start_time:,.0f}s")
