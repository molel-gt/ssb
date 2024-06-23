#!/usr/bin/env python3

import gmsh


def create_mesh(output_meshfile):
    micron = 1e-6
    resolution = 5 * micron
    LX = 150 * micron
    LY = 40 * micron
    points = [
        (0, 0, 0),
        (0.5 * LX, 0, 0),
        (LX, 0, 0),
        (LX, LY, 0),
        (0.5 * LX, LY, 0),
        (0, LY, 0),
    ]
    gpoints = []
    lines = []

    gmsh.initialize()
    gmsh.model.add('two-subdomains')
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    for idx, p in enumerate(points):
        gpoints.append(
            gmsh.model.occ.addPoint(*p)
        )
    gmsh.model.occ.synchronize()
    gmsh.model.occ.synchronize()
    for idx in range(0, len(points)-1):
        lines.append(
            gmsh.model.occ.addLine(gpoints[idx], gpoints[idx+1])
        )
    lines.append(
        gmsh.model.occ.addLine(gpoints[-1], gpoints[0])
    )
    lines.append(
        gmsh.model.occ.addLine(gpoints[1], gpoints[4])
    )
    phase_1 = 1
    phase_2 = 2

    left = 1
    bottom_left = 2
    bottom_right = 3
    right = 4
    top_right = 5
    top_left = 6
    middle = 7
    labels = {
        "phase_1": phase_1,
        "phase_2": phase_2,
        "left": left,
        "bottom_left": bottom_left,
        "bottom_right": bottom_right,
        "right": right,
        "top_right": top_right,
        "top_left": top_left,
        "middle": middle,
    }

    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(1, [lines[-2]], left, "left")
    gmsh.model.addPhysicalGroup(1, [lines[2]], right, "right")
    gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [-1]], middle, "middle")
    gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [0]], bottom_left, "bottom left")
    gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [4]], top_left, "top left")
    gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [1]], bottom_right, "bottom right")
    gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [3]], top_right, "top right")

    gmsh.model.occ.synchronize()
    se_loop = gmsh.model.occ.addCurveLoop([lines[idx] for idx in [0, 6, 4, 5]])
    pe_loop = gmsh.model.occ.addCurveLoop([lines[idx] for idx in [1, 2, 3, 6]])
    gmsh.model.occ.synchronize()
    se_phase = gmsh.model.occ.addPlaneSurface([se_loop])
    pe_phase = gmsh.model.occ.addPlaneSurface([pe_loop])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [se_phase], markers.phase_1, "phase 1")
    gmsh.model.addPhysicalGroup(2, [pe_phase], markers.phase_2, "phase 2")
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.write(output_meshfile)
    gmsh.finalize()

    return labels
