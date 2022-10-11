# Estimation of Effective Properties Using Finite Element Analysis
Two Python models are provided that estimate the effective conductivity and the effective active material specific area. We focus on the contact area between active material and solid electrolyte.
The [constants](constants.py) file contains constant variables such as default bulk ionic conductivity, which we set at 0.1 S/m (1 mS/cm).
For finite element analysis, we use dolfinx ([`FEniCSx`](https://github.com/FEniCS/dolfinx)). There are many installation options, but we found conda to work best for PC, e.g. in Ubuntu OS. In HPC clusters, installation via spack is preferred. We reproduce the process for installing dolfinx via conda:
```
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
```
Other required Python packages are specified in the [requirements](requirements.txt) file and can be installed via:
```
pip3 install -r requirements.txt --user
```
## Geometry Preparation
The geometry used in our models is prepared from segmented tomograms. For ease of communication, we label our segmented phases as 0 for void, 1 for solid electrolyte and 2 for active material. The volume reconstruction from segmented images is achieved using the [volume_reconstruction](volume_reconstruction.py) program. The parameters are:
```
(base) molel@molel-ThinkPad-T15-Gen-2i ~/dev/ssb (dev) $ python3 volume_reconstruction.py -h
usage: volume_reconstruction.py [-h] --img_folder IMG_FOLDER --grid_info GRID_INFO [--origin ORIGIN] [--resolution [RESOLUTION]] [--phase [PHASE]] [--scale_x [SCALE_X]] [--scale_y [SCALE_Y]]
                                [--scale_z [SCALE_Z]]

Reconstructs volume from segemented images.

optional arguments:
  -h, --help            show this help message and exit
  --img_folder IMG_FOLDER
                        Directory with input .tif files
  --grid_info GRID_INFO
                        Grid size is given by Nx-Ny-Nz such that the lengths are (Nx-1) by (Ny - 1) by (Nz - 1).
  --origin ORIGIN       Where to select choice grid from available segmented image array such that `subdata = data[origin_x:Nx, origin_y:Ny, origin_z:Nz]`
  --resolution [RESOLUTION]
                        Minimum resolution using gmsh
  --phase [PHASE]       Phase that we want to reconstruct, e.g. 0 for void, 1 for solid electrolyte and 2 for active material
  --scale_x [SCALE_X]   Value to scale the Lx grid size given to match dimensions of mesh files.
  --scale_y [SCALE_Y]   Value to scale the Ly grid size given to match dimensions of mesh files.
  --scale_z [SCALE_Z]   Value to scale the Lz grid size given to match dimensions of mesh files.
```

The volume reconstruction proceeds as follows:
- Load the segmented stack of images into a 3D array
- Select coordinates of voxels that belong to phase 1 (solid electrolyte) and label them using natural numbers starting from 0
- Select coordinates of voxels that belong to phase 1 (solid electrolyte), and are neighbors to phase 2 (active material), e.g. `(x_0, y_0, z_0)` and `(x_0, y_0 + 1, z_0)` are neighbors, but  `(x_0, y_0, z_0)` and `(x_0, y_0 + 1, z_0 + 1)` are not neighbors
- Save the effective electrolyte coordinates to the file `effective_electrolyte.pickle` within the input data directory
- Create an ordered tuple of neighboring solid electrolyte voxels that form a tetrahedra such that for 8 adjacent voxels that form a cube we end up with 5 non-intersecting tetrahedra.
- Optionally refine the tetrahedra using [TetGen](https://wias-berlin.de/software/index.jsp?id=TetGen&lang=1#Introduction) with inputs points in a `.node` file and input tetrahedra in a `.ele` file
- Optionally refine the mesh further using [GMSH](https://gmsh.info/#Documentation)
- Create a tetrahedral mesh file called tetr.xdmf that is scaled to match the voxel volume size
- Using [ParaView](https://www.paraview.org/), extract the external surface of the tetrahedral mesh file&emdash;tetr.xdmf
- Label surfaces of the above surface mesh that are spanned by the coordinates of the effective electrolyte obtained in step 3, and save the triangle mesh to tria.xmdf
## Estimation of Effective Conductivity
This is achieved using the [transport](transport.py) model. The model input parameters are:
```
(base) molel@molel-ThinkPad-T15-Gen-2i ~/dev/ssb (dev) $ python3 transport.py -h
usage: transport.py [-h] --grid_size GRID_SIZE --data_dir DATA_DIR [--voltage [VOLTAGE]] [--scale_x [SCALE_X]] [--scale_y [SCALE_Y]] [--scale_z [SCALE_Z]] [--loglevel [LOGLEVEL]]

Estimates Effective Conductivity.

optional arguments:
  -h, --help            show this help message and exit
  --grid_size GRID_SIZE
                        Lx-Ly-Lz
  --data_dir DATA_DIR   Directory with tria.xdmf and tetr.xdmf mesh files. Output files potential.xdmf and current.xdmf will be saved here
  --voltage [VOLTAGE]   Potential to set at the left current collector. Right current collector is set to a potential of 0
  --scale_x [SCALE_X]   Value to scale the Lx grid size given to match dimensions of mesh files.
  --scale_y [SCALE_Y]   Value to scale the Ly grid size given to match dimensions of mesh files.
  --scale_z [SCALE_Z]   Value to scale the Lz grid size given to match dimensions of mesh files.
  --loglevel [LOGLEVEL]
                        Logging level, e.g. ERROR, DEBUG, INFO, WARNING
```
The output units for area and volume are assumed to be in micrometers while effective conductivity is reported in Siemens per meter.
## Estimation of Effective Active Material Specific Area
This is achieved using the [effective active material area](effective_active_material_area.py) model. The model input parameters are:
```
(base) molel@molel-ThinkPad-T15-Gen-2i ~/dev/ssb (dev) $ python3 effective_active_material_area.py -h
usage: effective_active_material_area.py [-h] --grid_size GRID_SIZE --data_dir DATA_DIR [--scale_x [SCALE_X]] [--scale_y [SCALE_Y]] [--scale_z [SCALE_Z]]

Estimates Effective Active Material Specific Area.

optional arguments:
  -h, --help            show this help message and exit
  --grid_size GRID_SIZE
                        Lx-Ly-Lz
  --data_dir DATA_DIR   Directory with tria.xdmf and tetr.xdmf mesh files, and effective_electrolyte.pickle file. Output files potential.xdmf and current.xdf will be saved here.
  --scale_x [SCALE_X]   Value to scale the Lx grid size given to match dimensions of mesh files.
  --scale_y [SCALE_Y]   Value to scale the Ly grid size given to match dimensions of mesh files.
  --scale_z [SCALE_Z]   Value to scale the Lz grid size given to match dimensions of mesh files.
```
The input meshfiles include two options:
- Tetrahedral mesh of the solid electrolyte phase (tetr.xdmf) and a pickle file *effective_electrolyte.pickle* containing coordinates of solid electrolyte that are in contact with active material
- Tetrahedral mesh of the solid electrolyte phase (tetr.xdmf), and a triangle mesh of the surface of solid electrolyte (tria.xmdf) with effective active material area labelled