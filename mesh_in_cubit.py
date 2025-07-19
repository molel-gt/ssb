import sys

import numpy as np

sys.path.append("/opt/Coreform-Cubit-2025.3/bin")
import cubit

last_id = 0
LX = 20
LY = 20
lxs = np.arange(-0.5*LX+2.5, 0.5*LX, 5)
lys = np.arange(-0.5*LY+2.5, 0.5*LY, 5)
for x in lxs:
    for y in lys:
        cubit.cmd(f"create cylinder radius 1.5 height 50")
        last_id += 1
        cubit.cmd(f"volume {last_id} move x {x} y {y} z 50")

cubit.cmd("create brick x 20 y 20 z 5")
last_id += 1
cubit.cmd(f"volume {last_id} move z 75")
cubit.cmd("unite all")
pos_am = cubit.parse_cubit_list('volume', 'all')
cubit.cmd("create brick x 20 y 20 z 75")
last_id = cubit.parse_cubit_list('volume', 'all')[-1]
cubit.cmd(f"volume {last_id} move z 37.5")
volume_ids = [v for v in cubit.parse_cubit_list('volume', 'all') if v not in pos_am]
# cubit.cmd(f"chop vol {volume_ids[0]} with vol {pos_am[0]} keep")
# cubit.cmd(f"subtract vol {pos_am[0]} from vol {volume_ids[0]} keep")
cubit.cmd(f"remove overlap volume {pos_am[0]} {volume_ids[0]} modify larger")
cubit.cmd("vol all scheme tetmesh")
cubit.cmd("mesh volume all")
filename = "mesh.bdf"
cubit.cmd(f'export nastran {filename} overwrite')
