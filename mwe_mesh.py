#!/usr/bin/env python3

import gmsh

class Markers:
    def __init__(self):
        pass

    @property
    def phase_1(self):
        return 1

    @property
    def phase_2(self):
        return 2

    @property
    def left(self):
        return 3

    @property
    def bottom_left(self):
        return 4

    @property
    def bottom_right(self):
        return 5

    @property
    def right(self):
        return 6

    @property
    def top_right(self):
        return 7

    @property
    def top_left(self):
        return 8

    @property
    def middle(self):
        return 9

    @property
    def insulated(self):
        return 10


if __name__ == '__main__':
    micron = 1e-6
    resolution = 5 * micron
    LX = 150 * micron
    LY = 40 * micron
    output_meshfile = 'mesh.msh'

    markers = Markers()
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

    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(1, [lines[-2]], markers.left, "left")
    gmsh.model.addPhysicalGroup(1, [lines[2]], markers.right, "right")
    gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [-1]], markers.middle, "middle")
    gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [0]], markers.bottom_left, "bottom left")
    gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [4]], markers.top_left, "top left")
    gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [1]], markers.bottom_right, "bottom right")
    gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [3]], markers.top_right, "top right")
    gmsh.model.addPhysicalGroup(1, [lines[idx] for idx in [0, 1, 3, 4]], markers.insulated, "insulated")

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
