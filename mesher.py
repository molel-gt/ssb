#!/usr/bin/env python3
import argparse
import math
import os
import sys

import gmsh
import numpy as np
import pymeshlab
import trimesh

import commons, utils

SCALING = [1, 1, 1]#[0.0858e-6, 0.0858e-6, 0.05e-6]


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
    args = parser.parse_args()
    scaling = [float(v) for v in args.scale.split(",")]
    markers = commons.Markers()
    workdir = os.path.join(f"output/segmentation/{args.phase}/{args.size}/{args.origin}")
    utils.make_dir_if_missing(workdir)
    x0, y0, z0 = [int(val) for val in args.origin.split("-")]
    # LX, LY, LZ = [int(val) for val in args.size.split("-")]
    # L_sep = args.L_sep * SCALING[0]
    # Lx = (LX-10) * SCALING[0]
    # Ly = (LY+5) * SCALING[1]
    # Lz = (LZ+5) * SCALING[2]
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
    gmsh.merge(f"output/segmentation/cam.1.vtk")
    # gmsh.merge("cam.msh")
    # gmsh.model.geo.synchronize()
    vols = gmsh.model.getEntities(3)
    # gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.mesh.setOrder(1)
    # gmsh.option.setNumber('Geometry.Tolerance', 1e-8)
    # gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.01)
    # gmsh.option.setNumber('Mesh.Optimize', 1)
    gmsh.option.setNumber('Mesh.Algorithm', 5)
    # gmsh.model.mesh.removeDuplicateNodes()
    # angle = 180/180. * np.pi
    angle = gmsh.onelab.getNumber('Parameters/Angle for surface detection')[0]
    forceParametrizablePatches = gmsh.onelab.getNumber(
        'Parameters/Create surfaces guaranteed to be parametrizable')[0]
    curveAngle = 180
    gmsh.model.mesh.createTopology()
    gmsh.model.mesh.classifySurfaces(angle * math.pi/180., True, forceParametrizablePatches, curveAngle * math.pi/180.)
    #gmsh.model.mesh.createGeometry()
    
    surfaces_adjacencies = []

    for i, entity in enumerate(gmsh.model.getEntities(1)):
        adj = gmsh.model.get_adjacencies(entity[0], entity[1])
        surfaces_adjacencies.append(adj[0])

    # Python function to group surfacs that share at least a single upward adjency
    surfaces_to_combine = group_surfaces_adjacencies(surfaces_adjacencies)

    # Create a list with the surface loops of each aggregate
    agg_surf_loop_list = []
    phase_volumes = [v[1] for v in gmsh.model.getEntities(3)]
    print(phase_volumes)
    for i, stc in enumerate(surfaces_to_combine):
        agg_surf = gmsh.model.geo.addSurfaceLoop(stc)   # Add the surface loop
        agg_surf_loop_list.append(agg_surf)             # Include in the list

    gmsh.model.geo.synchronize()
    vols = gmsh.model.getEntities(3)
    print(vols)
    surfs = gmsh.model.getEntities(2)
    
    lxs = []
    lys = []
    lzs = []
    for surf in surfs:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.get_bounding_box(*surf)
        lxs.extend([xmin, xmax])
        lys.extend([ymin, ymax])
        lzs.extend([zmin, zmax])
    tol = 1.0 * SCALING[0]
    lx = np.max(lxs) + tol
    ly = np.max(lys) + tol
    lz = np.max(lzs) + tol
    gmsh.model.geo.removeAllDuplicates()
    gmsh.model.geo.synchronize()
    surf_loops = []
    print(np.min(lxs), np.max(lxs), Lx)
    print(np.min(lys), np.max(lys), Ly)
    print(np.min(lzs), np.max(lzs), Lz)
    sloop = create_box_surface_loop(Lx=Lx, Ly=Ly+padding, Lz=Lz+padding, L_sep=L_sep, origin=(-padding, -padding, -padding))
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(3, phase_volumes, markers.positive_am, "CAM")
    matrix_volume = gmsh.model.geo.addVolume([sloop] + agg_surf_loop_list)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(3, [matrix_volume], markers.electrolyte, "Electrolyte")
    gmsh.model.geo.synchronize()
    phys_vols = gmsh.model.getPhysicalGroups(3)

    left_surfs = []
    right_surfs = []
    interface_surfs = []
    insulated_am = []
    insulated_se = []
    surfs = gmsh.model.getEntities(2)
    print(np.min(lxs))
    for surf in surfs:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.get_bounding_box(*surf)
        if np.isclose(xmin, 0, atol=1e-8) and np.isclose(xmax, 0, atol=1e-8):
            right_surfs.append(surf[1])
            print(np.isclose(xmin, 0, atol=1e-7), np.isclose(xmax, 0, atol=1e-7))
        # if np.isclose(xmin, 0, atol=1e-6) and np.isclose(xmax, 0, atol=1e-6):
        #     right_surfs.append(surf[1])
        elif np.isclose(xmin, lx + L_sep, atol=1e-7) and np.isclose(xmin, lx + L_sep, atol=1e-7):
            left_surfs.append(surf[1])
        elif np.isclose(ymin, ymax) and (np.isclose(ymin, np.min(lxs)) or np.isclose(ymin, ly)):
            insulated_am.append(surf[1])
        elif np.isclose(zmin, zmax) and (np.isclose(zmin, 0) or np.isclose(zmin, lz)):
            insulated_am.append(surf[1])
        else:
            # if surf[1] in agg_surf_loop_list:
            interface_surfs.append(surf[1])
            # else:
            #     print(surf[1])
    gmsh.model.addPhysicalGroup(2, left_surfs, markers.left, "Left")
    gmsh.model.addPhysicalGroup(2, right_surfs, markers.right, "Right")
    gmsh.model.addPhysicalGroup(2, interface_surfs, markers.electrolyte_v_positive_am, "SE/AM")

    gmsh.model.geo.synchronize()

    gmsh.write("mesh.geo_unrolled")
    gmsh.model.mesh.generate()
    gmsh.write(os.path.join(workdir, "mesh.msh"))
