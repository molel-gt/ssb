#!/usr/bin/env python3
import csv
import os
import subprocess

import gmsh
import meshio
import numpy as np

import commons, geometry, utils


markers = commons.SurfaceMarkers()
phases = commons.Phases()
CELL_TYPES = commons.CellTypes()

# meshing
resolution = 0.1
scale_factor = [100e-6, 100e-6, 0]
program = 'gitt_geometry.geo'  # 'semicircle.geo'  # 'gitt.geo'



def create_geometry(relative_radius, outdir, scale_factor=scale_factor):
    utils.make_dir_if_missing(outdir)
    gmsh.initialize()
    gmsh.model.add("gitt")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1e-3)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)
    cloop1 = gmsh.model.occ.addRectangle(0, 0, 0, 10, 10)
    cloop2 = gmsh.model.occ.addCurveLoop([gmsh.model.occ.addCircle(5, 5, 0, relative_radius * 5)])
    gmsh.model.occ.synchronize()
    surf = gmsh.model.occ.addPlaneSurface([cloop1, cloop2])
    gmsh.model.occ.synchronize()
    lines = gmsh.model.getEntities(1)
    right_ccs = []
    insulateds = []
    left_ccs = []
    msh_fpath = os.path.join(outdir, "mesh.msh")
    for line in lines:
        com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
        if np.isclose(com[0], 10):
            insulateds.append(line[1])
        elif np.isclose(com[0], 0):
            left_ccs.append(line[1])
        elif np.isclose(com[1], 10) or np.isclose(com[1], 0):
            insulateds.append(line[1])
        else:
            right_ccs.append(line[1])
    left_cc = gmsh.model.addPhysicalGroup(1, left_ccs, markers.left_cc)
    gmsh.model.setPhysicalName(1, left_cc, "left_cc")
    right_cc = gmsh.model.addPhysicalGroup(1, right_ccs, markers.right_cc)
    gmsh.model.setPhysicalName(1, right_cc, "right_cc")
    insulated = gmsh.model.addPhysicalGroup(1, insulateds, markers.insulated)
    gmsh.model.setPhysicalName(1, insulated, "insulated")
    gmsh.model.addPhysicalGroup(2, [surf], 1)
    gmsh.model.mesh.generate(2)
    gmsh.write(msh_fpath)
    gmsh.finalize()
    
    msh = meshio.read(f"{msh_fpath}")
    tria_xdmf_unscaled = geometry.create_mesh(msh, CELL_TYPES.triangle)
    tria_xdmf_unscaled.write(os.path.join(outdir, "tria_unscaled.xdmf"))
    tria_xdmf_scaled = geometry.scale_mesh(tria_xdmf_unscaled, CELL_TYPES.triangle, scale_factor=scale_factor)
    tria_xdmf_scaled.write(os.path.join(outdir, "tria.xdmf"))
    line_xdmf_unscaled = geometry.create_mesh(msh, "line")
    line_xdmf_unscaled.write(os.path.join(outdir, "line_unscaled.xdmf"))
    line_xdmf_scaled = geometry.scale_mesh(line_xdmf_unscaled, CELL_TYPES.line, scale_factor=scale_factor)
    line_xdmf_scaled.write(os.path.join(outdir, "line.xdmf"))


if __name__ == '__main__':
    with open('data/study_3.csv') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            print(f"Creating geometry for relative radius {row['relative_radius']}")
            create_geometry(float(row['relative_radius']), row['outdir'])
    