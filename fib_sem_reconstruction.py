#!/usr/bin/env python3
import argparse
import os
import glob

import gmsh
import matplotlib.pyplot as plt
import matplotlib as mpl
import meshio
import numpy as np
import pyvista as pv
import spam.DIC
import spam
import spam.plotting

from pymeshfix._meshfix import PyTMesh
from skimage.io import imread_collection
from skimage import morphology, measure

import commons, plot_opts, utils


plt.rcParams.update(plot_opts.params)

cam_dir = os.path.join(os.environ["WORK_DIR"], "output/segmentation/cam")
voids_dir = os.path.join(os.environ["WORK_DIR"], "output/segmentation/voids")


def group_surfaces_adjacencies(adj):
    '''
    Function to goup the surfaces adjacencies.
    Reference: https://stackoverflow.com/a/4842897
    '''
    l = adj
    out = []
    while len(l)>0:
        first, *rest = l
        first = set(first)

        lf = -1
        while len(first)>lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)     
            rest = rest2

        out.append(list(first))
        l = rest
    return out


def get_aggregates_and_write_to_file(tomo, phase="voids", data_shape=(500, 500, 202)):
    utils.make_dir_if_missing(f"output/segmentation/{phase}")
    utils.make_dir_if_missing(f"output/segmentation/{phase}/aggs")

    # binary segmentation of data
    print(f"binary segmentation of data of {phase}")
    binary_labels_spam = spam.label.watershed(tomo)
    radii = spam.label.equivalentRadii(binary_labels_spam)
    radii_sieved = np.copy(radii)
    radii_sieved[radii_sieved<10] = 0

    # Create the image with each aggregate labelled
    spam_sieved = spam.label.convertLabelToFloat(binary_labels_spam, radii_sieved)

    # Get new labels for sieved aggregates
    spam_sieved_labels = spam.label.watershed(spam_sieved)


    # fig, axs = plt.subplots(1, 3, figsize=(10, 4), dpi=150)

    # axs[0].imshow(spam_sieved_labels[:, :, 10])
    # axs[1].imshow(spam_sieved_labels[:, spam_sieved_labels.shape[1]//2, :])
    # axs[2].imshow(spam_sieved_labels[:, :, spam_sieved_labels.shape[2]//2])

    # plt.show()


    # Converting the np.array image to a pyvista Uniform Grid
    print("Create pyvista uniform grid")
    pv_sieved = pv.ImageData()
    pv_sieved.dimensions = data_shape
    # pv_sieved.spacing = [160e-6, 160e-6, 160e-6]
    pv_sieved.origin = [0, 0, 0]
    pv_sieved.point_data['Label'] = spam_sieved_labels.T.flatten()
    agg_flag = np.zeros(spam_sieved_labels.T.shape).flatten()
    agg_flag[spam_sieved_labels.T.flatten()!=0] = 1
    pv_sieved.point_data['Aggs'] = agg_flag
    pv_sieved


    # Saving the aggregates to a vtk file
    pv_sieved.save(f'output/segmentation/{phase}/0_tomo.vtk')


    pv_sieved_e_d = pv_sieved.copy()
    print("Dilation and erosion to remove small features")
    n_ero_dil = 2
    ks = 5
    for i in range(n_ero_dil):
        for j in range(0, 40):
            pv_sieved_e_d = pv_sieved_e_d.image_dilate_erode(dilate_value=0, erode_value=j, kernel_size=(ks, ks, ks))
        
    for i in range(n_ero_dil + 1):
        for j in range(0, 40):
            pv_sieved_e_d = pv_sieved_e_d.image_dilate_erode(dilate_value=j, erode_value=0, kernel_size=(ks, ks, ks))


    # Saving the result to a vtk file
    pv_sieved_e_d.save(f'./output/segmentation/{phase}/1_tomo_e_d.vtk')


    # Creating a common flag for all aggregates to perform the surface meshing
    agg_flag_e_d = np.zeros(spam_sieved_labels.T.shape).flatten()
    agg_flag_e_d[pv_sieved_e_d.point_data['Label'] != 0] = 1
    pv_sieved.point_data['Aggs_e_d'] = agg_flag_e_d

    # Applying the marching cubes algorithm
    contour = pv_sieved_e_d.contour([1], scalars='Label', method='marching_cubes')

    # Saving the result to a vtk file
    contour.save('./output/2_contour_raw_mesh.vtk')


    # Creating the raw surface mesh
    surf_raw_mesh = contour.extract_geometry()

    # Filling the smaller holes using pyvista method
    surf_raw_mesh.fill_holes(5, inplace=True)

    # Using the clean functionality to remove degenerate surfaces etc
    surf_raw_mesh.clean(inplace=True)

    print("Repairing mesh using pymeshfix")
    mfix = PyTMesh(False)  # False removes extra verbose output
    mfix.load_array(surf_raw_mesh.points, surf_raw_mesh.faces.reshape((surf_raw_mesh.n_faces, 4))[:, 1:] )

    # Fills all the holes having at at most 'nbe' boundary edges. If
    # 'refine' is true, adds inner vertices to reproduce the sampling
    # density of the surroundings. Returns number of holes patched.  If
    # 'nbe' is 0 (default), all the holes are patched.
    mfix.fill_small_boundaries(refine=True)

    # Converting the pymeshfix object to pyvista polydata
    vert, faces = mfix.return_arrays()
    triangles = np.empty((faces.shape[0], 4), dtype=faces.dtype)
    triangles[:, -3:] = faces
    triangles[:, 0] = 3

    surf_raw_mesh = pv.PolyData(vert, triangles)


    # Splitting the aggregates
    aggs_raw = surf_raw_mesh.split_bodies(label=True)
    sieved_aggs = []

    # Performing the sieving based on the surface area of the aggregates
    for agg in aggs_raw:
        if agg.area > 25:
            sieved_aggs.append(agg)
            
    sieved_aggs_raw = pv.MultiBlock(sieved_aggs)


    # Performing smoothing of the aggregates and saving each one in an individual stl
    for i, sie_agg in enumerate(sieved_aggs):
        print(f'Getting Mesh for Agg: {i+1}')
        sie_agg_raw_surf = sie_agg.extract_geometry()
        sie_agg_smooth_surf = sie_agg_raw_surf.smooth_taubin(n_iter=20, pass_band=0.1, progress_bar=False)
        pv.save_meshio(f'./output/segmentation/{phase}/aggs/agg_{i+1}.stl', sie_agg_smooth_surf)

    # Saving it to a vtk file
    sieved_aggs_raw_surf = sieved_aggs_raw.extract_geometry()
    surf_smooth_mesh = sieved_aggs_raw_surf.smooth_taubin(n_iter=20, pass_band=0.1)
    surf_smooth_mesh.save(f'./output/segmentation/{phase}/3_surf_smooth_mesh.vtk')

    return


def create_volumes_from_stl(phase):
    # Merge each aggregate STL file
    for i, agg_path in enumerate(glob.glob(f'./output/segmentation/{phase}/aggs/*')):
        print(i)
        gmsh.merge(os.path.join(agg_path))

    # Split each surfaces for creating the separated geometry entities
    gmsh.model.mesh.classifySurfaces(gmsh.pi, True, True, gmsh.pi)



    # Create a geometry for each one of the discrete entities (aggregates)
    gmsh.model.mesh.createGeometry()


    # As the gmsh `classifySurfaces()` function splits the aggregates
    # surfaces into multiple parts (most of the time into two parts), retrieve 
    # which surfaces are part of which aggregates through adjencies of the curves
    surfaces_adjacencies = []
    for i, entity in enumerate(gmsh.model.getEntities(1)):
        surfaces_adjacencies.append(gmsh.model.get_adjacencies(entity[0], entity[1])[0])

    # Python function to group surfacs that share at least a single upward adjency
    surfaces_to_combine = group_surfaces_adjacencies(surfaces_adjacencies)


    # Create a list with the surface loops of each aggregate
    volumes = []
    agg_surf_loop_list = []
    for i, stc in enumerate(surfaces_to_combine):
        print(f"Processing surface {i} to volume")
        agg_surf = gmsh.model.geo.addSurfaceLoop(stc)   # Add the surface loop
        agg_surf_loop_list.append(agg_surf)             # Include in the list
        gmsh.model.geo.addVolume([agg_surf], tag=i)     # Create the volume
        volumes.append(i)
    return volumes, agg_surf_loop_list


def create_box_surface_loop(Lx, Ly, Lz, L_sep):
    coords = [
        (0, 0, 0),
        (Lx, 0, 0),
        (Lx, L_sep + Ly, 0),
        (0, L_sep + Ly, 0),
        (0, 0, Lz),
        (Lx, 0, Lz),
        (Lx, L_sep + Ly, Lz),
        (0, L_sep + Ly, Lz),
    ]
    points = [gmsh.model.geo.addPoint(*p) for p in coords]
    lines = [gmsh.model.geo.addLine(points[i], points[i+1]) for i in range(4-1)]
    lines.append(gmsh.model.geo.addLine(points[3], points[0])) # line 4
    lines.extend([gmsh.model.geo.addLine(points[i], points[i+1]) for i in range(4, 7)])
    lines.append(gmsh.model.geo.addLine(points[7], points[4])) # line 7
    lines.append(gmsh.model.geo.addLine(points[3], points[7])) # line 8
    lines.append(gmsh.model.geo.addLine(points[4], points[0])) # line 9
    lines.append(gmsh.model.geo.addLine(points[1], points[5])) # line 10
    lines.append(gmsh.model.geo.addLine(points[2], points[6])) # line 11
    loops = []
    gmsh.model.geo.synchronize()
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [0, 1, 2, 3]], reorient=True))
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [4, 5, 6, 7]], reorient=True))
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [8, 7, 9, 3]], reorient=True))
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [9, 0, 10, 4]], reorient=True))
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [10, 5, 11, 1]], reorient=True))
    loops.append(gmsh.model.geo.addCurveLoop([lines[idx] for idx in [11, 6, 8, 2]], reorient=True))
    gmsh.model.geo.synchronize()
    surfs = [gmsh.model.geo.addPlaneSurface([loop]) for loop in loops]
    gmsh.model.geo.synchronize()
    surf_loop = gmsh.model.geo.addSurfaceLoop(surfs)

    return surf_loop


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='secondary current distribution')
    parser.add_argument('--size', help='Lx-Ly-Lz', required=True, type=str)
    parser.add_argument("--origin", help="where to extract data", nargs='?', const=1, default='0-0-0', type=str)
    parser.add_argument("--phase", help="particulate phase", nargs='?', const=1, default='cam', type=str)
    parser.add_argument("--L_sep", help="separator thickness", nargs='?', const=1, default=100, type=float)
    args = parser.parse_args()

    phase = args.phase
    x0, y0, z0 = [int(val) for val in args.origin.split("-")]
    Lx, Ly, Lz = [int(val) for val in args.size.split("-")]
    markers = commons.Markers()
    tomo = np.zeros((Lx + 1, Ly + 1, Lz + 1), dtype=np.bool)
    tomo = tomo.astype(np.uint8)
    for img_id in range(1, Lz + 2):
        img_cam = plt.imread(os.path.join(cam_dir, f"{str(img_id).zfill(3)}.tif"))[:Lx+1, :Ly+1]
        # img_voids = plt.imread(os.path.join(voids_dir, f"{str(img_id).zfill(3)}.tif"))
        tomo[np.isclose(img_cam, 2), img_id - 1] = 1
        # tomo[np.isclose(img_voids, 1), img_id - 1] = 0
    get_aggregates_and_write_to_file(tomo, phase=phase, data_shape=tomo.shape)
    gmsh.initialize()  # Initialize the gmsh API
    phase_volumes, agg_surf_loop_list = create_volumes_from_stl(phase=phase)

    # Save the last tag index for the aggregate
    agg_last_idx = phase_volumes[-1]
    sloop = create_box_surface_loop(Lx=Lx, Ly=Ly, Lz=Lz, L_sep=args.L_sep)
    matrix_volume = gmsh.model.addVolume([sloop] + agg_surf_loop_list, tag=agg_last_idx + 1)

    # Synchronize the built-in CAD representation with the current Gmsh model
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(3, [matrix_volume], tag=markers.electrolyte)
    gmsh.model.addPhysicalGroup(3, phase_volumes, tag=markers.active_material)
    gmsh.model.geo.synchronize()
    # Selection of the Delaunay algorithm for meshing
    # gmsh.option.setNumber("Mesh.Algorithm3D", 10)

    # Creation of a distance field to control the mesh element sides
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "FacesList", [item for sublist in surfaces_to_combine for item in sublist])
    gmsh.model.mesh.field.setNumber(1, "NNodesByEdge", 10)

    # We then define a `Threshold' field, which uses the return value of the
    # `Distance' field 1 in order to define a simple change in element size
    # depending on the computed distances
    #
    # SizeMax -                     /------------------
    #                              /
    #                             /
    #                            /
    # SizeMin -o----------------/
    #          |                |    |
    #        Point         DistMin  DistMax
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", 1.0)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", 2.5)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 1)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 5)

    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.model.mesh.generate()
    # Writing the `.msh` file
    gmsh.write("./output/4_final_mesh.msh")
    gmsh.finalize()
