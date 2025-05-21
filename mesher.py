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


def create_box(Lx, Ly, Lz):
    """"""
    z0_points = [
        (0, 0, 0),
        (Lx, 0, 0),
        (Lx, Ly, 0),
        (0, Ly, 0),
    ]

    points0 = []
    lines = []

    for i in range(4):
        idx = gmsh.model.geo.addPoint(*z0_points[i])
        points0.append(idx)

    gmsh.model.geo.synchronize()
    for i in range(-1, 3):
        idx = gmsh.model.geo.addLine(points0[i], points0[i + 1])
        lines.append(
            idx
        )

    loops = [gmsh.model.geo.addCurveLoop(lines[:4])]

    gmsh.model.geo.synchronize()
    surface = gmsh.model.geo.addPlaneSurface(loops)
    box = gmsh.model.geo.extrude([(2, surface)], 0, 0, 1, heights=[Lz], recombine=True)
    gmsh.model.geo.synchronize()
    return [(3, box)]


if __name__ == '__main__':
    markers = commons.Markers()
    folder = sys.argv[1]

    gmsh.initialize()
    gmsh.merge(f"output/segmentation/{folder}-repaired.stl")
    gmsh.model.mesh.createTopology()
    gmsh.model.mesh.classifySurfaces(gmsh.pi, True, True, gmsh.pi)
    gmsh.model.mesh.createGeometry()

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
        # gmsh.model.geo.synchronize()
        
    # Save the last tag index for the aggregate
    # agg_last_idx = i

    # gmsh.model.geo.synchronize()
    # surfs = gmsh.model.getEntities(2)
    # surface_loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfs])
    # gmsh.model.geo.synchronize()
    # vols = gmsh.model.geo.addVolume([surface_loop])
    # gmsh.model.geo.synchronize()
    vols = gmsh.model.getEntities(3)
    # main_box = gmsh.model.geo.addBox(0, 0, 0, 500, 500, 202)
    print(vols)
    gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], markers.void, "volume")
    # gmsh.model.addPhysicalGroup(2, [v[1] for v in surfs], 1, "surface")
    gmsh.model.mesh.generate(3)
    gmsh.write(f"{folder}.msh")
