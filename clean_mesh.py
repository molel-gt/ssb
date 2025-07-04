import sys

import trimesh
import numpy as np
import matplotlib.pyplot as plt

from mayavi import mlab
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from skimage.draw import ellipsoid

folder = sys.argv[1]
nz = 21
step = 1
data3d = np.zeros((500, 500, nz), dtype=bool)
for idx in range(1, nz+1):
    img_file = f"output/segmentation/{folder}/{str(idx).zfill(3)}.tif"
    data = np.asarray(plt.imread(img_file))
    data3d[:, :, idx-1] = data
encoding =  trimesh.voxel.encoding.DenseEncoding(data3d)
voxels = trimesh.voxel.base.VoxelGrid(encoding)
print(voxels.volume)
mesh = voxels.marching_cubes
trimesh.exchange.export.export_mesh(mesh, f"output/segmentation/{folder}.stl")
