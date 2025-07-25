#!/usr/bin/env python3
import os
import random
import sys

import numpy as np

sys.path.append("/opt/Coreform-Cubit-2025.3/bin")
import cubit

import commons#, mesh_utils


resolution = 0.005


if __name__ == '__main__':
    mesh_folder = sys.argv[1]
    markers = commons.Markers()
    last_id = 0
    LX = 20
    LY = 20
    L_CELL = 80
    lxs = np.arange(-0.5*LX/L_CELL+2.5/L_CELL, 0.5*LX/L_CELL, 5/L_CELL)
    lys = np.arange(-0.5*LY/L_CELL+2.5/L_CELL, 0.5*LY/L_CELL, 5/L_CELL)
    radius = 1.5/L_CELL
    height = 55/L_CELL
    cubit.cmd("Set Quality Threshold 0.7")
    for x in lxs:
        for y in lys:
            cubit.cmd(f"create cylinder radius {radius} height {height}")
            last_id += 1
            cubit.cmd(f"volume {last_id} move x {x} y {y} z {52.5/L_CELL}")

    L_slab_am = 5

    z_pos = (L_CELL - L_slab_am)/L_CELL
    random.seed(0)
    L_SEP = 25
    while z_pos > 0.3175:
        for x in lxs:
            for y in lys:
                p_val = random.uniform(0, 1)
                if p_val <= 0.25:
                    p_val2 = random.uniform(0, 1)
                    r = 1.75 + p_val2 * 5.0
                    if (x + r/L_CELL) >= 0.5 * LX/L_CELL or (x - r/L_CELL) <= -0.5 * LX/L_CELL:
                        continue
                    if (y + r/L_CELL) >= 0.5 * LY/L_CELL or (y - r/L_CELL) <= -0.5 * LY/L_CELL:
                        continue
                    if (z_pos + r/L_CELL >= 1 - L_slab_am/L_CELL) or (z_pos -r/L_CELL) <= L_SEP/L_CELL:
                        continue
                    cubit.cmd(f"create sphere radius {r/L_CELL}")
                    last_id = cubit.get_last_id("volume")
                    cubit.cmd(f"volume {last_id} move x {x} y {y} z {z_pos}")
        z_pos -= 2.5/L_CELL

    cubit.cmd(f"create brick x {20/L_CELL} y {20/L_CELL} z {5/L_CELL}")
    last_id += 1
    cubit.cmd(f"volume {last_id} move z {77.5/L_CELL}")
    cubit.cmd("unite all")
    cubit.cmd(f"create brick x {20/L_CELL} y {20/L_CELL} z {80/L_CELL}")
    last_id = cubit.parse_cubit_list('volume', 'all')[-1]
    cubit.cmd(f"volume {last_id} move z {40/L_CELL}")
    volume_ids = cubit.parse_cubit_list('volume', 'all')
    cubit.cmd(f"remove overlap volume {volume_ids[0]} {volume_ids[1]} modify larger")
    curves = cubit.parse_cubit_list('curve', 'all')
    circle_arcs = []
    for curve in curves:
        # print(cubit.get_curve_center(curve))
        if np.isclose(cubit.get_arc_length(curve), 2 * np.pi * radius):
            circle_arcs.append(curve)
    cubit.cmd(f"modify curve {' '.join(map(str, circle_arcs))} blend radius 0.005")
    cubit.cmd("imprint all")
    cubit.cmd("merge all")
    volumes = cubit.parse_cubit_list('volume', 'all')
    cubit.cmd(f"volume {volumes[0]} name 'positive_am'")
    cubit.cmd(f"volume {volumes[1]} name 'solid_electrolyte'")
    cubit.cmd(f"block {markers.electrolyte} solid_electrolyte")
    cubit.cmd(f"block {markers.positive_am} positive_am")
    surfaces = cubit.parse_cubit_list('surface', 'all')
    insulated = []
    interface = []
    for surf in surfaces:
        centroid = cubit.get_surface_centroid(surf)
        area = cubit.get_surface_area(surf)
        if np.isclose(centroid[2], 0):
            cubit.cmd(f"surface {surf} name 'left_surf'")
            cubit.cmd(f"block {markers.left} left_surf")
        elif np.isclose(centroid[2], 80/L_CELL):
            cubit.cmd(f"surface {surf} name 'right_surf'")
            cubit.cmd(f"block {markers.right} right_surf")
        elif np.isclose(np.abs(centroid[0]), 10/L_CELL) or np.isclose(np.abs(centroid[1]), 10/L_CELL):
            insulated.append(surf)
        else:
            interface.append(surf)
    cubit.cmd("set developer commands on")
    cubit.cmd(f"block {markers.insulated} surface {' '.join(map(str, insulated))}")
    cubit.cmd(f"block {markers.electrolyte_v_positive_am} surface {' '.join(map(str, interface))}")
    # cubit.cmd(f"surface {' '.join(map(str, interface))} size {resolution}")
    cubit.cmd("surface all scheme trimesh")
    cubit.cmd("mesh surface all")
    cubit.cmd("vol all scheme tetmesh")
    cubit.cmd("mesh volume all")
    cubit.cmd(f"refine surface {' '.join(map(str, interface))} size 0.005 bias 1.2")

    filename = os.path.join(mesh_folder, "mesh.bdf")
    cubit.cmd(f"export nastran '{filename}' overwrite")
    # mesh_utils.convert_to_xdmf(filename, "tetra", "triangle", "nastran:ref")
