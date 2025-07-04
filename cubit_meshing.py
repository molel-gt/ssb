import sys

sys.path.append("/opt/Coreform-Cubit-2025.3/bin")
import cubit

cubit.cmd("import stl '1.stl' feature_angle 90.00 merge")
cubit.cmd("vol all scheme tet")
cubit.cmd("mesh vol all")
cubit.cmd("export nastran 'mesh.bdf' mesh_only overwrite everything")
