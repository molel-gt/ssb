#!/usr/bin/env python3
import sys

import gmsh
import trimesh
import math

folder = sys.argv[1]

gmsh.initialize()
# gmsh.merge(f"output/segmentation/{folder}.msh")
# surf_mesh = trimesh.interfaces.gmsh.load_gmsh(f"output/segmentation/{folder}.msh")
# print(surf_mesh)
# trimesh.exchange.export.export_mesh(surf_mesh, f"output/segmentation/{folder}-surf.stl")
# gmsh.merge(f"output/segmentation/{folder}-surf.stl")
gmsh.merge(f"output/segmentation/{folder}.stl")
# gmsh.model.mesh.createTopology()
gmsh.model.geo.synchronize()
gmsh.model.mesh.classifySurfaces(math.pi/3, boundary=True, forReparametrization=False, curveAngle=math.pi, exportDiscrete=False)
# gmsh.model.mesh.createGeometry()
gmsh.model.geo.synchronize()
# surfs = gmsh.model.getEntities(2)
# sloops = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfs])
# gmsh.model.geo.addVolume(sloops)
# gmsh.model.geo.synchronize()
# print(surfs)
vols = gmsh.model.getEntities(3)
print(vols)
gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], 1, "volume")
gmsh.model.addPhysicalGroup(2, [v[1] for v in surfs], 1, "surface")
surfs = gmsh.model.occ.getEntitiesInBoundingBox(0, 0, 0, 500, 500, 202)
print(len(surfs))

gmsh.model.mesh.generate(3)
gmsh.write(f"{folder}.msh")
