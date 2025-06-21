#!/usr/bin/python3
import os

import meshio
import numpy as np

import commons

if __name__ == '__main__':
    workdir = os.path.join(os.environ["HOME"], "OneDrive/PhD/Data/composite-electrode")
    # input_file = os.path.join(workdir, "merged.1.mesh")
    input_file = os.path.join(workdir, "post_cgalmesh.mesh")
    output_file = os.path.join(workdir, "mesh.mesh")
    markers = commons.Markers()
    mesh = meshio.read(input_file)
    points = mesh.points #/ np.max(mesh.points[:, 0])
    x_min = np.min(mesh.points[:, 0])
    x_max = np.max(mesh.points[:, 0])
    points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
    points[:, 1] = points[:, 1] / (x_max - x_min)
    points[:, 2] = points[:, 2] * 0.2e-06 / (0.08e-06 * (x_max - x_min))
    tets = mesh.get_cells_type("tetra")
    triangles = mesh.get_cells_type("triangle")
    lines = mesh.get_cells_type("line")

    tets_cell_data = mesh.get_cell_data("medit:ref", "tetra")
    tets_cell_data2 = np.zeros(tets_cell_data.shape, dtype=np.int32)
    tets_cell_data2[np.isclose(tets_cell_data, 2)] = markers.electrolyte
    tets_cell_data2[np.isclose(tets_cell_data, 3)] = markers.positive_am
    tets_cell_data = tets_cell_data2
    triangle_cell_data = mesh.get_cell_data("medit:ref", "triangle")

    # line_cell_data = mesh.get_cell_data("medit:ref", "line")

    electrolyte_tets = set(tets[np.isclose(tets_cell_data, markers.electrolyte), :].flatten().tolist())
    pos_am_tets = set(tets[np.isclose(tets_cell_data, markers.positive_am), :].flatten().tolist())
    interface = electrolyte_tets.intersection(pos_am_tets)
    facet_cell_data = np.zeros(triangle_cell_data.shape, dtype=np.int32)
    for idx, tria in enumerate(triangles):
        tria_set = set(tria.tolist())
        if tria_set.issubset(interface):
            facet_cell_data[idx] = markers.electrolyte_v_positive_am
        else:
            x_vals = [points[idx, 0] for idx in tria_set]
            if np.all(np.isclose(x_vals, 0, atol=1e-3)):
                facet_cell_data[idx] = markers.right
            elif np.all(np.isclose(x_vals, 1, atol=1e-3)):
                print("Left")
                facet_cell_data[idx] = markers.left
    cells = [
        ("tetra", tets),
        ("triangle", triangles),
        # ("line", lines)
        ]
    cells_data = {"medit:ref": [tets_cell_data, facet_cell_data]}
    out_mesh = meshio.Mesh(points=points,
                           cells=cells,
                           cell_data=cells_data
                           )
    out_mesh.write(output_file)