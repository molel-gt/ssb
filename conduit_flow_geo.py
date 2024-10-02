import argparse
import os
import gmsh
import numpy as np

import utils

class Labels:
    def __init__(self):
        pass

    @property
    def domain(self):
        return 1

    @property
    def inlet(self):
        return 1

    @property
    def outlet(self):
        return 2

    @property
    def inlet_outlet_separation(self):
        return 3

    @property
    def left(self):
        return 4

    @property
    def right(self):
        return 5

    @property
    def top(self):
        return 6

    @property
    def insulated(self):
        return 7



def create_mesh(output_meshfile, markers, L=1, w_over_L=0.1, h_over_L=0.1, resolution=0.1, refine=True):
    """
    """
    points_bottom = [
        (0, 0, 0),
        (w_over_L * L, 0, 0),
        (L * (1 - w_over_L), 0, 0),
        (L, 0, 0),
    ]
    points_top = [
        (0, h_over_L * L, 0),
        (L, h_over_L * L, 0)
    ]
    gpoints_top = []
    gpoints_bottom = []
    lines = []
    gmsh.initialize()
    gmsh.model.add('conduit')
    gpoints_top.append(gmsh.model.occ.addPoint(*points_top[0]))
    gpoints_top.append(gmsh.model.occ.addPoint(*points_top[1]))
    gpoints_bottom.extend([gmsh.model.occ.addPoint(*p) for p in points_bottom])

    lines.append(gmsh.model.occ.addLine(gpoints_bottom[0], gpoints_top[0]))
    lines.append(gmsh.model.occ.addLine(gpoints_top[0], gpoints_top[1]))
    lines.append(gmsh.model.occ.addLine(gpoints_top[1], gpoints_bottom[-1]))
    lines.append(gmsh.model.occ.addLine(gpoints_bottom[-1], gpoints_bottom[-2]))
    lines.append(gmsh.model.occ.addLine(gpoints_bottom[-2], gpoints_bottom[-3]))
    lines.append(gmsh.model.occ.addLine(gpoints_bottom[-3], gpoints_bottom[-4]))

    curve_loop = gmsh.model.occ.addCurveLoop(lines)
    surf = gmsh.model.occ.addPlaneSurface([curve_loop])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [surf], markers.domain, "domain")

    gmsh.model.addPhysicalGroup(1, [lines[0]], markers.left, "left")
    gmsh.model.addPhysicalGroup(1, [lines[1]], markers.top, "top")
    gmsh.model.addPhysicalGroup(1, [lines[2]], markers.right, "right")
    gmsh.model.addPhysicalGroup(1, [lines[3]], markers.outlet, "outlet")
    gmsh.model.addPhysicalGroup(1, [lines[4]], markers.inlet_outlet_separation, "inlet-outlet separation")
    gmsh.model.addPhysicalGroup(1, [lines[-1]], markers.inlet, "inlet")
    gmsh.model.addPhysicalGroup(1, lines[:-3] + [lines[-2], lines[0], lines[1], lines[2]], markers.insulated, "insulated")

    if refine:
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", lines)

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", L * resolution/5)
        gmsh.model.mesh.field.setNumber(2, "LcMax", L * resolution)
        gmsh.model.mesh.field.setNumber(2, "DistMin", L * resolution)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 5 * L * resolution)

        gmsh.model.mesh.field.add("Max", 5)
        gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(5)
        gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.write(output_meshfile)
    gmsh.finalize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='secondary current distribution')
    parser.add_argument('--mesh_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument("--Lc", help="characteristic length", nargs='?', const=1, default=1.0, type=np.float16)
    parser.add_argument("--h_over_L", help="aspect ratio", nargs='?', const=1, default=0.1, type=np.float16)
    parser.add_argument("--w_over_L", help="aspect ratio of inlet/outlet", nargs='?', const=1, default=0.1, type=np.float16)
    parser.add_argument("--resolution_lc", help="resolution relative to characteristic length", nargs='?', const=1, default=0.1, type=np.float16)
    args = parser.parse_args()
    mesh_folder = os.path.join("output", "conduit_flow")
    workdir = os.path.join(args.mesh_folder, str(args.Lc), str(args.h_over_L), str(args.w_over_L))
    utils.make_dir_if_missing(workdir)
    output_meshfile_path = os.path.join(workdir, "mesh.msh")
    markers = Labels()
    create_mesh(output_meshfile_path, markers, L=args.Lc, resolution=args.resolution_lc, h_over_L=args.h_over_L, w_over_L=args.w_over_L)
