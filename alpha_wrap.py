import pygalmesh
import meshio

myMesh = meshio.read("output/segmentation/cam/201-201-201/0-0-0/aggs/agg_1.stl")
mesh = pygalmesh.alpha_wrap_3(myMesh, 0.01, 0.1)
mesh.write("output/segmentation/cam/201-201-201/0-0-0/aggs/agg_1.ply")
