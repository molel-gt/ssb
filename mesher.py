#!/usr/bin/env python3
import gmsh

gmsh.initialize()
gmsh.merge("voids.msh")
gmsh.model.occ.synchronize()
vols = gmsh.model.getEntities(3)
print(vols)
gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], 1, "volume")
gmsh.model.mesh.generate(3)
gmsh.write("voids-clean.msh")
