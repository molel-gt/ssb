#!/usr/bin/env python3
import argparse
import math
import gmsh

def create_mesh(input_file):
    """"""
    gmsh.initialize()
    gmsh.model.add("Reconstruction")
    gmsh.merge(input_file)
    gmsh.option.setNumber("General.NumThreads", 8)
    angle = 90
    curveAngle = 180
    gmsh.model.mesh.classifySurfaces(angle * math.pi/180., True, True, curveAngle * math.pi/180.)
    # gmsh.model.mesh.createTopology()
    gmsh.model.geo.synchronize()
    vols = gmsh.model.getEntities(3)
    print(vols)
    surfs = gmsh.model.getEntities(2)
    print(surfs)
    gmsh.model.mesh.generate(3)
    gmsh.write(f"mesh.msh")
    gmsh.finalize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rescale output of cgalmesh")
    parser.add_argument("--input_file", "-i", help="input file", type=str, required=True)
    parser.add_argument("--output_file", "-o", help="output file", type=str, required=True)
    parser.add_argument("--scale", "-s", help="scaling factor in sx,sy,sz", type=str, required=True)
    parser.add_argument('--normalize_direction', help='normalize direction', nargs='?',
                        const=1, default=0, type=int)
    parser.add_argument("--remesh_only", help="whether to remesh only", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    # if args.remesh_only:
    #     create_mesh(args.output_file)
    #     quit()
    scale_x, scale_y, scale_z = [float(s) for s in args.scale.split(",")]

    nodes = []
    max_x = 1
    max_y = 1
    max_z = 1
    idx = 0
    max_idx = 0
    with open(args.input_file, "r") as f_in:
        for idx, row in enumerate(f_in.readlines()):
            if idx < 3:
                continue
            if idx == 3:
                max_idx = int(row)
            if row.strip("\n") == "Triangles":
                break
            values = row.split(" ")
            
            if 3 < idx < max_idx + 4:
                values[:3] = [float(v) for v in values[:3]]
                values[-1] = values[-1].strip("\n")
                if values[0] > max_x:
                    max_x = values[0]
                if values[1] > max_y:
                    max_y = values[1]
                if values[2] > max_z:
                    max_z = values[2]
                nodes.append(values)

    scaling_value = 1
    if args.normalize_direction == 0:
        scaling_value = max_x * scale_x
    elif args.normalize_direction == 1:
        scaling_value = max_y * scale_y
    else:
        scaling_value = max_z * scale_z

    with open(args.output_file, "w") as f_out:
        f_out.write("MeshVersionFormatted 1\n")
        f_out.write("Dimension 3\n")
        f_out.write("Vertices\n")
        f_out.write(f"{max_idx}\n")
        for (x, y, z, tag) in nodes:
            f_out.write(f"{x*scale_x/scaling_value} {y*scale_y/scaling_value} {z*scale_z/scaling_value} {tag}\n")

        with open(args.input_file, "r") as f_in:
            start_writing = False
            for row in f_in.readlines():
                if row.strip("\n") == "Triangles":
                    start_writing = True
                if start_writing:
                    f_out.write(row)
    create_mesh(args.output_file)
