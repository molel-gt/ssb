import os
import meshio
import numpy as np


data_dir = "/home/lesh/dev/ssb/"

msh = meshio.read(os.path.join(data_dir, "porous.msh"))
points = msh.points,
cells = msh.cells
cell_data = msh.cell_data
field_data = msh.field_data


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points,
                           cells={cell_type: cells},
                           cell_data={"name_to_read": [cell_data]}
                           )
    if prune_z:
        out_mesh.prune_z_0()
    return out_mesh


line_mesh = create_mesh(msh, "line")
meshio.write("mesh_line.xdmf", line_mesh)

triangle_mesh = create_mesh(msh, "triangle")
meshio.write("mesh_tria.xdmf", triangle_mesh)

tetra_mesh = create_mesh(msh, "tetra")
meshio.write("mesh_tetra.xdmf", tetra_mesh)
