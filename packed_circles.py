#! /usr/bin/env python3
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

import argparse
import gmsh
import meshio
import numpy as np

import commons, geometry


markers = commons.SurfaceMarkers()
CELL_TYPES = commons.CellTypes()
scale_factor = [100e-6, 100e-6, 0]


def read_circles_position_file(circles_position_path):
    """
    Reads file input that contains the centers of spheres. The path is assumed to have
    been generated using code from Skoge et al.
    """
    centers = []
    radius = 0
    n = 0
    with open(circles_position_path) as fp:
        for i, row in enumerate(fp.readlines()):
            if i < 2:
                continue
            if i == 2:
                n = int(row)
            if i == 3:
                radius = float(row)
            if i < 6:
                continue
            x, y, _ = row.split(' ')
            centers.append((float(x), float(y), 0))
    return centers, float(radius)/2, n


def build_packed_circles_mesh(output_mesh_file, circles_locations_files):
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1e-6)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1e-2)
    gmsh.model.add("GITT")
    Lx, Ly, Lz = 1, 1, 1
    resolution = 1e-7
    channel0 = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    # channel1 = gmsh.model.occ.addRectangle(0, 0, 0, 0.05, Ly)
    # channel1 = gmsh.model.occ.addPlaneSurface([channel1])
    # channel2 = gmsh.model.occ.addRectangle(Lx - 0.05, 0, 0, 0.05, Ly)
    # channel2 =  gmsh.model.occ.addCurveLoop([channel2])
    # channel0 = gmsh.model.occ.addPlaneSurface([channel0])
    gmsh.model.occ.synchronize()
    circles = []
    for circles_locations_file in circles_locations_files:
        centers, r, n_circles = read_circles_position_file(circles_locations_file)
        for center in centers:
            x, y, z = center
            circle = gmsh.model.occ.addCircle(*center, r)
            cloop =  gmsh.model.occ.addCurveLoop([circle])
            surf = gmsh.model.occ.addPlaneSurface([cloop])
            circles.append(surf)
            gmsh.model.occ.synchronize()
    gmsh.model.occ.synchronize()
    channel = gmsh.model.occ.cut([(2, channel0)], [(2, piece) for piece in circles])

    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=2)
    marker = 11
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], marker)
    gmsh.model.setPhysicalName(volumes[0][0], marker, "conductor")
    lines = gmsh.model.occ.getEntities(dim=1)
    walls = []
    left_ccs = []
    right_ccs = []
    for line in lines:
        com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
        if np.isclose(com[0], Lx):
            right_ccs.append(line[1])
        elif np.isclose(com[0], 0):
            walls.append(line[1])
        elif np.isclose(com[1], 0) or np.isclose(com[1], Ly):
            walls.append(line[1])
        else:
            left_ccs.append(line[1])
    left_cc = gmsh.model.addPhysicalGroup(1, left_ccs, markers.left_cc)
    gmsh.model.setPhysicalName(1, left_cc, "left_cc")
    right_cc = gmsh.model.addPhysicalGroup(1, right_ccs, markers.right_cc)
    gmsh.model.setPhysicalName(1, right_cc, "right_cc")
    insulated = gmsh.model.addPhysicalGroup(1, walls, markers.insulated)
    gmsh.model.setPhysicalName(1, insulated, "insulated")

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "FacesList", [insulated, left_cc, right_cc])

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", resolution)
    gmsh.model.mesh.field.setNumber(2, "LcMax", 20*resolution)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.1)

    # gmsh.model.mesh.field.add("Min", 5)
    # gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
    # gmsh.model.mesh.field.setAsBackgroundMesh(5)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    
    gmsh.write(output_mesh_file)
    gmsh.finalize()
    
    return


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Effective Conductivity of Packed Spheres.')
    parser.add_argument('--pf', help="Packing fraction (percent)", default=30)
    args = parser.parse_args()
    pf = args.pf
    grid_size = '1-1-1'
    spheres_locations_files = ['mesh/circles/round-1.dat']#, 'mesh/circles/round-2.dat', 'mesh/circles/round-3.dat']  #f"mesh/packed_spheres/{grid_size}_{pf}/{pf}.dat"
    output_mesh_file = f"mesh/circles/spheres.msh"
    
    build_packed_circles_mesh(output_mesh_file, spheres_locations_files)
    msh = meshio.read(output_mesh_file)
    tria_xdmf_unscaled = geometry.create_mesh(msh, CELL_TYPES.triangle)
    line_xdmf_unscaled = geometry.create_mesh(msh, "line")
    tria_xdmf_unscaled.write("mesh/circles/tria_unscaled.xdmf")
    tria_xdmf_scaled = geometry.scale_mesh(tria_xdmf_unscaled, CELL_TYPES.triangle, scale_factor=scale_factor)
    tria_xdmf_scaled.write("mesh/circles/tria.xdmf")
    line_xdmf_unscaled.write("mesh/circles/line_unscaled.xdmf")
    line_xdmf_scaled = geometry.scale_mesh(line_xdmf_unscaled, CELL_TYPES.line, scale_factor=scale_factor)
    line_xdmf_scaled.write("mesh/circles/line.xdmf")
    print(f"Wrote tria.xdmf and line.xdmf mesh files to directory: mesh/circles")