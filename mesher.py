#!/usr/bin/env python3
import sys

import gmsh
import trimesh

import commons

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
    markers = commons.Markers()
    folder = sys.argv[1]
    L_sep = 100
    Lx = 499
    Ly = 499
    Lz = 201
    gmsh.initialize()
    gmsh.merge(f"output/segmentation/voids.msh")
    gmsh.model.geo.synchronize()
    vols = gmsh.model.getEntities(3)
    print(vols)
    gmsh.model.mesh.createTopology()
    # gmsh.model.mesh.classifySurfaces(gmsh.pi, True, True, gmsh.pi)
    # gmsh.model.mesh.createGeometry()
    gmsh.model.geo.synchronize()
    surfs = gmsh.model.getEntities(2)

    surfaces_adjacencies = []
    for i, entity in enumerate(gmsh.model.getEntities(1)):
        surfaces_adjacencies.append(gmsh.model.get_adjacencies(entity[0], entity[1])[0])

    # Python function to group surfacs that share at least a single upward adjency
    surfaces_to_combine = group_surfaces_adjacencies(surfaces_adjacencies)

    # Create a list with the surface loops of each aggregate
    agg_surf_loop_list = []
    for i, stc in enumerate(surfaces_to_combine):
        agg_surf = gmsh.model.geo.addSurfaceLoop(stc)   # Add the surface loop
        agg_surf_loop_list.append(agg_surf)             # Include in the list
        gmsh.model.geo.addVolume([agg_surf], tag=i)     # Create the volume
        gmsh.model.geo.synchronize()
        
    # Save the last tag index for the aggregate
    agg_last_idx = i
    surfs = gmsh.model.getEntities(2)
    sloop = create_box_surface_loop(Lx=Lx, Ly=Ly, Lz=Lz, L_sep=args.L_sep)
    matrix_volume = gmsh.model.geo.addVolume([sloop] + agg_surf_loop_list, tag=agg_last_idx + 1)
    vols = gmsh.model.getEntities(3)
    print(vols)
    gmsh.model.geo.synchronize()
    # gmsh.model.addPhysicalGroup(3, [matrix_volume], tag=markers.electrolyte)
    gmsh.model.addPhysicalGroup(3, phase_volumes, tag=markers.positive_am)
    gmsh.model.geo.synchronize()
    # gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], markers.void, "volume")
    # gmsh.model.addPhysicalGroup(2, [v[1] for v in surfs], 1, "surface")
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(f"{folder}.msh")
