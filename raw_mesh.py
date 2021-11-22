import pygmsh
import gmsh
import os
import numpy as np


file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh/100_0_100", "sphere-in-box.msh")

with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_min = 5
    geom.characteristic_length_max = 5
    box1 = geom.add_box([0, 0, 0], [100, 100, 100])
    ball1 = geom.add_ball([10, 50, 50], 15)
    ball2 = geom.add_ball([50, 50, 50], 15)
    ball3 = geom.add_ball([90, 50, 50], 15)
    final = geom.boolean_difference(box1, geom.boolean_union([ball1, ball2, ball3]))
    geom.add_physical(final, "box")
    geom.synchronize()
    mesh = geom.generate_mesh(dim=3)
    gmsh.write(file_path)
