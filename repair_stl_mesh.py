#!/usr/bin/env python3
import sys
import pymeshlab


if __name__ == '__main__':
    meshfile_path = sys.argv[1]
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(meshfile_path)
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_re_orient_faces_coherently()
    ms.meshing_remove_folded_faces()
    ms.meshing_remove_null_faces()
    ms.compute_selection_by_self_intersections_per_face()
    ms.meshing_remove_selected_vertices_and_faces()
    ms.compute_selection_by_non_manifold_edges_per_face()
    ms.compute_selection_by_non_manifold_per_vertex()
    #ms.meshing_snap_mismatched_borders()
    ms.meshing_close_holes()
    ms.save_current_mesh(meshfile_path)
