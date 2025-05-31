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
    parser.add_argument("--phase", help="particulate phase", nargs='?', const=1, default='cam', type=str)
    parser.add_argument("--L_sep", help="separator thickness", nargs='?', const=1, default=100, type=float)
    args = parser.parse_args()
    markers = commons.Markers()
    workdir = os.path.join(f"output/segmentation/{args.phase}/{args.size}/{args.origin}")
    

    # mesh = mrmeshpy.loadMesh("output/segmentation/cam.stl")
    # params = mrmeshpy.FixMeshDegeneraciesParams()
    # params.maxDeviation = 1e-5 * mesh.computeBoundingBox().diagonal()
    # params.tinyEdgeLength = 1e-3
    # mrmeshpy.fixMeshDegeneracies(mesh, params)
    # # Find single edge for each hole in mesh
    # hole_edges = mesh.topology.findHoleRepresentiveEdges()

    # for e in hole_edges:
    #     #  Setup filling parameters
    #     params = mrmeshpy.FillHoleParams()
    #     params.metric = mrmeshpy.getUniversalMetric(mesh)
    #     #  Fill hole represented by `e`
    #     mrmeshpy.fillHole(mesh, e, params)
    # mrmeshpy.saveMesh(mesh, "output/segmentation/cam-repaired.stl")
    # ms = pymeshlab.MeshSet()
    # ms.load_new_mesh("output/segmentation/cam.stl")
    # ms.meshing_remove_unreferenced_vertices()
    # ms.meshing_repair_non_manifold_vertices()
    # ms.meshing_repair_non_manifold_edges()
    # ms.meshing_close_holes()
    # ms.meshing_snap_mismatched_borders()
    # # ms.generate_resampled_uniform_mesh()
    # ms.save_current_mesh("output/segmentation/cam-repaired.stl")
    utils.make_dir_if_missing(workdir)
    x0, y0, z0 = [int(val) for val in args.origin.split("-")]
    LX, LY, LZ = [int(val) for val in args.size.split("-")]
    L_sep = args.L_sep * SCALING[0]
    Lx = (LX-10) * SCALING[0]
    Ly = (LY+5) * SCALING[1]
    Lz = (LZ+5) * SCALING[2]
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
    # sloop = create_box_surface_loop(Lx=Lx, Ly=Ly, Lz=Lz, L_sep=args.L_sep)
    # gmsh.model.geo.synchronize()
    # matrix_volume = gmsh.model.geo.addVolume([sloop] + agg_surf_loop_list)
    # gmsh.model.geo.synchronize()
    # gmsh.model.geo.synchronize()
    # for idx in range(2, 3):
    #     gmsh.merge(f"output/segmentation/cam/200-200-200/0-0-0/aggs/agg_{idx}.stl")
    gmsh.merge(f"output/segmentation/cam.1.vtk")
    # gmsh.merge("cam.msh")
    # gmsh.model.geo.synchronize()
    vols = gmsh.model.getEntities(3)
    # gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.mesh.setOrder(1)
    # gmsh.option.setNumber('Geometry.Tolerance', 1e-8)
    # gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.01)
    # gmsh.option.setNumber('Mesh.Optimize', 1)
    # gmsh.option.setNumber('Mesh.Algorithm', 5)
    # gmsh.model.mesh.removeDuplicateNodes()
    # angle = 180/180. * np.pi
    angle = gmsh.onelab.getNumber('Parameters/Angle for surface detection')[0]
    forceParametrizablePatches = gmsh.onelab.getNumber(
        'Parameters/Create surfaces guaranteed to be parametrizable')[0]
    curveAngle = 180
    gmsh.model.mesh.createTopology()
    gmsh.model.mesh.classifySurfaces(angle * math.pi/180., True, forceParametrizablePatches, curveAngle * math.pi/180.)
    gmsh.model.mesh.createGeometry()
    
    surfaces_adjacencies = []

    for i, entity in enumerate(gmsh.model.getEntities(1)):
        adj = gmsh.model.get_adjacencies(entity[0], entity[1])
        surfaces_adjacencies.append(adj[0])

    # Python function to group surfacs that share at least a single upward adjency
    surfaces_to_combine = group_surfaces_adjacencies(surfaces_adjacencies)

    # Create a list with the surface loops of each aggregate
    agg_surf_loop_list = []
    phase_volumes = []
    for i, stc in enumerate(surfaces_to_combine):
        agg_surf = gmsh.model.geo.addSurfaceLoop(stc)   # Add the surface loop
        agg_surf_loop_list.append(agg_surf)             # Include in the list
        gmsh.model.geo.addVolume([agg_surf], tag=i)     # Create the volume
        # phase_volumes.append(i)
    gmsh.model.geo.synchronize()
    vols = gmsh.model.getEntities(3)
    bndry = [s[1] for s in gmsh.model.getBoundary(vols)]
    surfaces_adjacencies = []

    for i, entity in enumerate(gmsh.model.getEntities(1)):
        adj = gmsh.model.get_adjacencies(entity[0], entity[1])
        surfs_new = [surf for surf in adj[0] if surf in bndry]
        # if adj[0] in bndry:
        surfaces_adjacencies.append(surfs_new)

    # Python function to group surfacs that share at least a single upward adjency
    surfaces_to_combine = group_surfaces_adjacencies(surfaces_adjacencies)

    # Create a list with the surface loops of each aggregate
    agg_surf_loop_list = []
    phase_volumes = []
    for i, stc in enumerate(surfaces_to_combine):
        agg_surf = gmsh.model.geo.addSurfaceLoop(stc)   # Add the surface loop
        agg_surf_loop_list.append(agg_surf) 
    # gmsh.model.geo.removeAllDuplicates()
    # gmsh.model.addPhysicalGroup(3, phase_volumes, tag=markers.positive_am)
    # gmsh.model.geo.synchronize()
    # print(phase_volumes, agg_surf_loop_list)
    # Save the last tag index for the aggregate
    # agg_last_idx = phase_volumes[-1]
    surfs = gmsh.model.getEntities(2)
    
    lxs = []
    lys = []
    lzs = []
    for surf in surfs:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.get_bounding_box(*surf)
        lxs.extend([xmin, xmax])
        lys.extend([ymin, ymax])
        lzs.extend([zmin, zmax])
    tol = 1e-8
    lx = np.max(lxs) #+ tol
    ly = np.max(lys) #+ tol
    lz = np.max(lzs) #+ tol
    # gmsh.model.geo.remove([(3, agg_last_idx+1)])
    # gmsh.model.geo.dilate(gmsh.model.getEntities(0)+gmsh.model.getEntities(1)+gmsh.model.getEntities(2)+gmsh.model.getEntities(3), 1, 1, 1, 0.0858e-6, 0.0858e-6, 0.05e-6)
    # gmsh.model.geo.synchronize()
    gmsh.model.geo.removeAllDuplicates()
    # gmsh.model.geo.synchronize()
    surf_loops = []
    # print(vols)
    # quit()

    sloop = create_box_surface_loop(Lx=lx, Ly=ly, Lz=lz, L_sep=L_sep, origin=(10*SCALING[0], np.min(lys)-tol, np.min(lzs)-tol))
    gmsh.model.geo.synchronize()
    # hole = gmsh.model.geo.addSurfaceLoop(bndry)
    # print(agg_surf_loop_list, hole)
    gmsh.model.addPhysicalGroup(3, phase_volumes, markers.positive_am, "CAM")
    # sloop = create_box_surface_loop(Lx=Lx, Ly=Ly, Lz=Lz, L_sep=L_sep)
    # gmsh.model.geo.synchronize()
    # print(sloop)
    matrix_volume = gmsh.model.geo.addVolume([sloop] + agg_surf_loop_list)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(3, [matrix_volume], markers.electrolyte, "Electrolyte")
    gmsh.model.geo.synchronize()
    vols = gmsh.model.getPhysicalGroups(3)
    print(vols)

    left_surfs = []
    right_surfs = []
    interface_surfs = []
    insulated_am = []
    insulated_se = []
    # lxs = []
    # lys = []
    # lzs = []
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
