#!/usr/bin/env python3

import gmsh
import numpy as np

gmsh.initialize()
gmsh.model.add('test_merge')

# Merge the stl file
gmsh.merge('output/segmentation/cam.stl')

angle = 10 * np.pi / 180.
curve_angle = 180 * np.pi / 180.
gmsh.model.mesh.classifySurfaces(
    angle=angle, 
    boundary=True, 
    forReparametrization=True, 
    curveAngle=curve_angle
    )

gmsh.option.setNumber("General.AbortOnError", 1)

# Reparameterize
gmsh.model.mesh.createGeometry()

# Generate the 3D mesh
gmsh.model.mesh.generate(3)

# Launch the GUI to see the results:
gmsh.fltk.run()
