#!/usr/bin/env python3
import argparse
import os
import sys

import gmsh
import meshlib.mrmeshpy as mrmeshpy
import numpy as np
import pymeshlab
import trimesh

import commons, utils

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


def create_box_surface_loop(Lx, Ly, Lz, L_sep):
    coords = [
        (0, 0, 0),
        (Lx, 0, 0),
        (Lx, L_sep + Ly, 0),
        (0, L_sep + Ly, 0),
        (0, 0, Lz),
        (Lx, 0, Lz),
        (Lx, L_sep + Ly, Lz),
        (0, L_sep + Ly, Lz),
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
    gmsh.model.geo.synchronize()
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [0, 1, 2, 3]], reorient=True))
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [4, 5, 6, 7]], reorient=True))
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [8, 7, 9, 3]], reorient=True))
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [9, 0, 10, 4]], reorient=True))
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [10, 5, 11, 1]], reorient=True))
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [11, 6, 8, 2]], reorient=True))
    gmsh.model.geo.synchronize()
    surfs = [gmsh.model.geo.addPlaneSurface([loop]) for loop in loops]
    gmsh.model.geo.synchronize()
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
    Lx, Ly, Lz = [int(val) for val in args.size.split("-")]
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    sloop = create_box_surface_loop(Lx=Lx, Ly=Ly, Lz=Lz, L_sep=args.L_sep)
    gmsh.model.geo.synchronize()
    gmsh.merge("output/segmentation/cam/aggs/agg_1.stl")
    # gmsh.model.geo.synchronize()
    # gmsh.model.mesh.createTopology()
    gmsh.model.geo.synchronize()
    vols = gmsh.model.getEntities(3)
    angle = 10.0/180.0 * np.pi
    curveAngle = 1.0 * np.pi
    gmsh.model.mesh.classifySurfaces(angle, True, True, curveAngle)
    gmsh.model.mesh.createGeometry()
    gmsh.model.geo.synchronize()
    surfs = gmsh.model.getEntities(2)
    surfaces_adjacencies = []
    gmsh.model.geo.synchronize()
    print(gmsh.model.getEntities(1))

    for i, entity in enumerate(gmsh.model.getEntities(1)):
        surfaces_adjacencies.append(gmsh.model.get_adjacencies(entity[0], entity[1])[0])

    # Python function to group surfacs that share at least a single upward adjency
    surfaces_to_combine = group_surfaces_adjacencies(surfaces_adjacencies)

    # Create a list with the surface loops of each aggregate
    agg_surf_loop_list = []
    phase_volumes = []
    for i, stc in enumerate(surfaces_to_combine):
        agg_surf = gmsh.model.geo.addSurfaceLoop(stc)   # Add the surface loop
        agg_surf_loop_list.append(agg_surf)             # Include in the list
        gmsh.model.geo.addVolume([agg_surf], tag=i)     # Create the volume
        phase_volumes.append(i)
        gmsh.model.geo.synchronize()
    # Save the last tag index for the aggregate
    agg_last_idx = phase_volumes[-1]
    surfs = gmsh.model.getEntities(2)
    vols = gmsh.model.getEntities(3)

    sloop = create_box_surface_loop(Lx=Lx, Ly=Ly, Lz=Lz, L_sep=args.L_sep)
    gmsh.model.geo.synchronize()
    matrix_volume = gmsh.model.geo.addVolume([sloop] + agg_surf_loop_list, tag=agg_last_idx + 1)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(3, [matrix_volume], tag=markers.electrolyte)
    # gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], tag=markers.positive_am)
    gmsh.model.geo.synchronize()

    left_surfs = []
    right_surfs = []
    interface_surfs = []
    insulated_am = []
    insulated_se = []
    for surf in surfs:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.get_bounding_box(*surf)
        if np.isclose(ymin, ymax) and np.isclose(ymin, 0):
            right_surfs.append(surf[1])
        elif np.isclose(ymin, ymax) and np.isclose(ymin, Ly + args.L_sep):
            left_surfs.append(surf[1])
        elif np.isclose(xmin, xmax) and (np.isclose(xmin, 0) or np.isclose(xmin, Lx)):
            insulated_am.append(surf[1])
        elif np.isclose(zmin, zmax) and (np.isclose(zmin, 0) or np.isclose(zmin, Lz)):
            insulated_am.append(surf[1])
        else:
            # if surf[1] in agg_surf_loop_list:
            interface_surfs.append(surf[1])
            # else:
            #     print(surf[1])
    # gmsh.model.addPhysicalGroup(2, left_surfs, markers.left, "Left")
    # gmsh.model.addPhysicalGroup(2, right_surfs, markers.right, "Right")
    # gmsh.model.addPhysicalGroup(2, interface_surfs, markers.electrolyte_v_positive_am, "SE/AM")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate()
    gmsh.write(os.path.join(workdir, "mesh.msh"))
