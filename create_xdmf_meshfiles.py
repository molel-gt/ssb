import os
import sys
import meshio
import numpy as np




def create_mesh(mesh, cell_type, prune_z=False):
    """
    """
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points,
                           cells={cell_type: cells},
                           cell_data={"name_to_read": [cell_data]}
                           )
    if prune_z:
        out_mesh.prune_z_0()
    return out_mesh


if __name__ == '__main__':
    input_meshfile = sys.argv[1]
    msh = meshio.read(input_meshfile)

    line_mesh_path = os.path.join(os.path.dirname(input_meshfile),
                                  "mesh_line.xdmf")
    tria_mesh_path = os.path.join(os.path.dirname(input_meshfile),
                                  "mesh_tria.xdmf")
    tetr_mesh_path = os.path.join(os.path.dirname(input_meshfile),
                                  "mesh_tetr.xdmf")
    line_mesh = create_mesh(msh, "line")
    meshio.write(line_mesh_path, line_mesh)

    triangle_mesh = create_mesh(msh, "triangle")
    meshio.write(tria_mesh_path, triangle_mesh)

    tetra_mesh = create_mesh(msh, "tetra")
    meshio.write(tetr_mesh_path, tetra_mesh)
