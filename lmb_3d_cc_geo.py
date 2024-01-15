#!/usr/env/bin python3
import argparse
import os

import gmsh
import meshio
import numpy as np

import configs, commons, geometry, utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Current Distribution')
    parser.add_argument("--name_of_study", help="name_of_study", nargs='?', const=1, default="lmb_3d_cc")
    parser.add_argument('--dimensions', help='integer representation of Lx-Ly-Lz of the grid', required=True)
    parser.add_argument("--resolution", help="maximum resolution in microns", nargs='?', const=1, default=1, type=int)
    parser.add_argument('--scaling', help='scaling key in `configs.cfg` to ensure geometry in meters', nargs='?',
                        const=1, default='MICRON_TO_METER', type=str)

    args = parser.parse_args()

    markers = commons.Markers()

    LX, LY, LZ = [int(val) for val in args.dimensions.split("-")]
    micron = 1e-6
    L_sep = 0.25 * LZ * micron
    L_neg_cc = 0.20 * LZ * micron
    L_sep_neg_cc = 0.15 * LZ * micron
    feature_radius = 0.05 * LY * micron
    disk_radius = LX * micron
    L_total = L_sep + L_neg_cc

    mesh_folder = os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], args.name_of_study, args.dimensions, str(args.resolution))
    output_meshfile = os.path.join(mesh_folder, 'mesh.msh')
    tetr_meshfile = os.path.join(mesh_folder, "tetr.xdmf")
    tria_meshfile = os.path.join(mesh_folder, "tria.xdmf")
    utils.make_dir_if_missing(mesh_folder)

    gmsh.initialize()
    gmsh.model.add('lithium-metal-leb')

    neg_cc = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L_neg_cc, disk_radius)
    gmsh.model.occ.synchronize()
    sep_main = gmsh.model.occ.addCylinder(0, 0, L_neg_cc, 0, 0, L_sep, disk_radius)
    gmsh.model.occ.synchronize()
    sep_neg_cc = gmsh.model.occ.addCylinder(0, 0, L_neg_cc - L_sep_neg_cc, 0, 0, L_neg_cc, feature_radius)
    gmsh.model.occ.synchronize()
    current_collector = gmsh.model.occ.cut([(3, neg_cc)], [(3, sep_neg_cc)], removeTool=False)
    gmsh.model.occ.synchronize()
    electrolyte = gmsh.model.occ.fuse([(3, sep_main)], [(3, sep_neg_cc)])
    gmsh.model.occ.synchronize()
    
    
    volumes = gmsh.model.occ.getEntities(3)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [volumes[1][1]], markers.electrolyte)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [volumes[0][1]], markers.negative_cc)
    gmsh.model.occ.synchronize()
    surfaces = gmsh.model.occ.getEntities(2)
    left = []
    right = []
    middle = []
    insulated = []
    insulated_ncc = []
    insulated_sep = []
    for surf in surfaces:
        com = gmsh.model.occ.getCenterOfMass(surf[0], surf[1])
        if np.isclose(com[2], 0):
            left.append(surf[1])
        elif np.isclose(com[2], L_total):
            right.append(surf[1])
        elif np.isclose(com[2], L_total - 0.5 * L_sep) or np.isclose(com[2], 0.5 * L_neg_cc):
            if np.isclose(com[2], 0.5 * L_neg_cc):
                insulated_ncc.append(surf[1])
            elif np.isclose(com[2], L_total - 0.5 * L_sep):
                insulated_sep.append(surf[1])
        else:
            middle.append(surf[1])
    insulated = insulated_ncc + insulated_sep
    lefttag = gmsh.model.addPhysicalGroup(2, left, markers.left)
    gmsh.model.occ.synchronize()
    righttag = gmsh.model.addPhysicalGroup(2, right, markers.right)
    gmsh.model.occ.synchronize()
    middletag = gmsh.model.addPhysicalGroup(2, middle, markers.middle)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, insulated_ncc, markers.insulated_negative_cc)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, insulated_sep, markers.insulated_electrolyte)
    gmsh.model.occ.synchronize()
    insulatedtag = gmsh.model.addPhysicalGroup(2, insulated, markers.insulated)
    gmsh.model.occ.synchronize()

    # refinement
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "FacesList", [insulatedtag, lefttag, middletag, righttag])

    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "LcMin", args.resolution * micron)
    gmsh.model.mesh.field.setNumber(threshold, "LcMax", 10 * args.resolution * micron)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.5 * micron)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", 5 * micron)

    max_f = gmsh.model.mesh.field.add("Max")
    gmsh.model.mesh.field.setNumbers(max_f, "FieldsList", [threshold])
    gmsh.model.mesh.field.setAsBackgroundMesh(max_f)
    gmsh.model.mesh.generate(3)
    gmsh.write(output_meshfile)
    gmsh.finalize()
    
    mesh_3d = meshio.read(output_meshfile)
    tetr_mesh = geometry.create_mesh(mesh_3d, "tetra")
    meshio.write(tetr_meshfile, tetr_mesh)
    tria_mesh = geometry.create_mesh(mesh_3d, "triangle")
    meshio.write(tria_meshfile, tria_mesh)
    print(f"Finished writing {tetr_meshfile} and {tria_meshfile}")
