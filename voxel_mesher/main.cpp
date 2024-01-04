#include <algorithm> 
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
// #include <H5Cpp.h>
#include <hdf5.h>
#include <iostream>
#include <map>
#include <ranges>
#include <vector>

#define TETR_FILE "tetr.h5"
#define TRIA_FILE "tria.h5"

namespace po = boost::program_options;
namespace fs = boost::filesystem;

bool is_boundary_point(std::map<std::vector<int>, int> all_points, std::vector<int> check_point){
    int num_neighbors = 0;
    int i, j, k;
    i = check_point[0];
    j = check_point[1];
    k = check_point[2];

    std::vector<std::vector<int>> neighbor_points = {
        {i + 1, j, k},
        {i, j + 1, k},
        {i, j, k + 1},
        {i - 1, j, k},
        {i, j - 1, k},
        {i, j, k - 1},
    };
    for (int idx = 0; idx < 6; idx++){
        if (all_points.count(neighbor_points[idx]) > 0){
            num_neighbors++;
        }
    }
    return num_neighbors != 6;
}

std::vector<int> get_tetrahedron(std::map<int, int> cube_points, int tet_number){
    std::vector<int> tet;
    int invalid = -1;
    switch(tet_number){
        case 0:
        {
            tet = {cube_points[0], cube_points[1], cube_points[3], cube_points[4]};
            break;
        }
        case 1:
        {
            tet = {cube_points[1], cube_points[2], cube_points[3], cube_points[6]};
            break;
        }
        case 2:
        {
            tet = {cube_points[4], cube_points[5], cube_points[6], cube_points[1]};
            break;
        }
        case 3:
        {
            tet = {cube_points[4], cube_points[7], cube_points[6], cube_points[3]};
            break;
        }
        case 4:
        {
            tet = {cube_points[4], cube_points[6], cube_points[1], cube_points[3]};
            break;
        }
        default:
        {
            tet = {invalid, invalid, invalid, invalid};
            break;
        }
    }

    std::sort(tet.begin(), tet.end());
    if (std::find(tet.begin(), tet.end(), -1) == tet.end()){
        std::cout << tet[0] << "," << tet[1] << "," << tet[2] << "," << tet[3] << "\n";
}
    return tet;
}

std::vector<std::vector<int>> get_tetrahedron_faces(std::map<int, int> local_cube_points, int tet_number){
    std::vector<std::vector<int>> local_tet_faces;
    switch (tet_number){
        case 0:
        {
            local_tet_faces = {
                {local_cube_points[0], local_cube_points[1], local_cube_points[4]},
                {local_cube_points[0], local_cube_points[3], local_cube_points[4]},
                {local_cube_points[1], local_cube_points[0], local_cube_points[3]},
                {local_cube_points[4], local_cube_points[1], local_cube_points[3]},
            };
            break;
        }
        case 1:
        {
            local_tet_faces = {
                {local_cube_points[1], local_cube_points[2], local_cube_points[6]},
                {local_cube_points[6], local_cube_points[2], local_cube_points[3]},
                {local_cube_points[1], local_cube_points[2], local_cube_points[3]},
                {local_cube_points[1], local_cube_points[6], local_cube_points[3]},
            };
            break;
        }
        case 2:
        {
            local_tet_faces = {
                {local_cube_points[1], local_cube_points[5], local_cube_points[4]},
                {local_cube_points[1], local_cube_points[6], local_cube_points[5]},
                {local_cube_points[4], local_cube_points[5], local_cube_points[6]},
                {local_cube_points[4], local_cube_points[6], local_cube_points[1]},
            };
            break;
        }
        case 3:
        {
            local_tet_faces = {
                {local_cube_points[4], local_cube_points[7], local_cube_points[3]},
                {local_cube_points[3], local_cube_points[7], local_cube_points[6]},
                {local_cube_points[4], local_cube_points[7], local_cube_points[6]},
                {local_cube_points[4], local_cube_points[6], local_cube_points[3]},
            };
            break;
        }
        case 4:
        {
            local_tet_faces = {
                {local_cube_points[4], local_cube_points[3], local_cube_points[1]},
                {local_cube_points[1], local_cube_points[3], local_cube_points[6]},
                {local_cube_points[4], local_cube_points[1], local_cube_points[6]},
                {local_cube_points[4], local_cube_points[3], local_cube_points[6]},
            };
            break;
        }
    }
    return local_tet_faces;
}

std::map<std::vector<int>, int> build_points_from_voxels(std::map<std::vector<int>, int> voxels, int phase, int Nx, int Ny, int Nz){
    std::map<std::vector<int>, int> output_points;
    int num_points = 0;
    for (int i = 0; i < Nx; i++){
        for (int j = 0; j < Ny; j++){
            for (int k = 0; k < Nz; k++){
                if (voxels.contains({i, j, k})){
                    if (voxels.at({i, j, k}) == phase){
                        output_points[{i, j, k}] = num_points++;
                    }
                }
            }
        }
    }

    return output_points;

}

std::vector<std::vector<int>> make_cube(int x, int y, int z){
    return {
            {x, y, z},
            {x + 1, y, z},
            {x + 1, y + 1, z},
            {x, y + 1, z},
            {x, y, z + 1},
            {x + 1, y, z + 1},
            {x + 1, y + 1, z + 1},
            {x, y + 1, z + 1}
        };
}

std::vector<int> remap_tetrahedrons(std::vector<int> tet, std::map<int, int> remap_dict){
    return {remap_dict.at(tet[0]), remap_dict.at(tet[1]), remap_dict.at(tet[2]), remap_dict.at(tet[3])};
}

std::vector<std::vector<int>> remap_tetrahedron_faces(std::vector<std::vector<int>> tet_faces, std::map<int, int> remap_dict){
    return {
        {remap_dict.at(tet_faces[0][0]), remap_dict.at(tet_faces[0][1]), remap_dict.at(tet_faces[0][2]), remap_dict.at(tet_faces[0][3])},
        {remap_dict.at(tet_faces[1][0]), remap_dict.at(tet_faces[1][1]), remap_dict.at(tet_faces[1][2]), remap_dict.at(tet_faces[1][3])},
        {remap_dict.at(tet_faces[2][0]), remap_dict.at(tet_faces[2][1]), remap_dict.at(tet_faces[2][2]), remap_dict.at(tet_faces[2][3])},
        {remap_dict.at(tet_faces[3][0]), remap_dict.at(tet_faces[3][1]), remap_dict.at(tet_faces[3][2]), remap_dict.at(tet_faces[3][3])},
    };

}

int main(int argc, char* argv[]){
    fs::path mesh_folder_path;
    int phase;
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Creates tetrahedral and triangle mesh in xdmf format from input voxels")
    ("mesh_folder_path,MESH_FOLDER_PATH", po::value<fs::path>(&mesh_folder_path)->required(),  "mesh folder path")
    ("phase,PHASE", po::value<int>(&phase)->required(),  "phase to reconstruct volume for")
    ("boundary_layer", po::value<bool>()->default_value(false), "whether or not to add half pixel boundary layer");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }
    mesh_folder_path = vm["mesh_folder_path"].as<fs::path>();
    bool boundary_layer = vm["boundary_layer"].as<bool>();
    phase = vm["phase"].as<int>();

    if (!boundary_layer){
        std::cout << "No boundary layer is written\n";
    }

    int Nx = 2, Ny = 2, Nz = 2;
    std::map<std::vector<int>, int> voxels;
    std::map<std::vector<int>, int> points;

    voxels[{0, 0, 0}] = 1;
    voxels[{1, 0, 0}] = 1;
    voxels[{1, 1, 0}] = 1;
    voxels[{0, 1, 0}] = 0;
    voxels[{0, 0, 1}] = 1;
    voxels[{1, 0, 1}] = 1;
    voxels[{1, 1, 1}] = 1;
    voxels[{0, 1, 1}] = 1;
    points = build_points_from_voxels(voxels, phase, Nx, Ny, Nz);

    // build tetrahedrons and faces
    std::vector<std::vector<int>> tetrahedrons;
    std::vector<std::vector<std::vector<int>>> tetrahedrons_faces;
    int invalid = -1;
    // build tetrahedrons -- these are refined to remove
    // reference to voxel coordinates that are not composing
    // any tetrahedron
    for (int i = 0; i < Nx; i++){
        for (int j = 0; j < Ny; j++){
            for (int k = 0; k < Nz; k++){
                std::vector<std::vector<int>> cube = make_cube(i, j, k);
                std::map<int, int> cube_points;
                for (int idx = 0; idx < 8; idx++){
                    std::vector<int> key = {cube[idx][0], cube[idx][1], cube[idx][2]};
                    if (points.contains(key)){
                        cube_points[idx] = points.at(key);
                    } else {
                        cube_points[idx] = invalid;
                    }
                }
                for (int idx = 0; idx < 5; idx++){
                    std::vector<int> tet = get_tetrahedron(cube_points, idx);
                    if (std::find(tet.begin(), tet.end(), invalid) == tet.end()){
                        std::cout << tet[0] << tet[1] << tet[2] << tet[3] << "\n";
                        tetrahedrons.push_back(tet);
                        std::vector<std::vector<int>> tet_faces = get_tetrahedron_faces(cube_points, idx);
                        tetrahedrons_faces.push_back(tet_faces);
                    }
                }
            }
        }
    }

    // points with <key,value> inverted to <value,key>
    std::map<int, std::vector<int>> points_inverse;
    std::vector<int> key;
    std::map<int, int> points_id_remapping;
    std::map<int, std::vector<int>> points_remapped;

    for (int i = 0; i < Nx; i++){
        for (int j = 0; j < Ny; j++){
            for (int k = 0; k < Nz; k++){
                key = {i, j, k};
                if (points.contains(key)){
                    points_inverse[points.at(key)] = key;
                }
            }
        }
    }
    // remap points
    int num_points = 0;
    int n_tets = tetrahedrons.size() / tetrahedrons[0].size();
    for (int idx = 0; idx < n_tets; idx++){
        std::vector<int> tet_points = tetrahedrons[idx];
        for (auto tet_point: tet_points){
            std::cout << tet_point << "\n";
            if (!points_id_remapping.contains(tet_point)){
                points_id_remapping[tet_point] = num_points;
                std::cout << tet_point << "\n";
                points_remapped[num_points] = points_inverse.at(tet_point);
                num_points ++;
            }
        }
    }
    for (int idx = 0; idx < n_tets; idx++){
        tetrahedrons[idx] = remap_tetrahedrons(tetrahedrons[idx], points_id_remapping);
        tetrahedrons_faces[idx] = remap_tetrahedron_faces(tetrahedrons_faces[idx], points_id_remapping);
    }

    // write hdf5 file
    std::vector<std::vector<int>> final_points;
    for (int idx = 0; idx < num_points; idx++){
        std::cout << points_remapped[idx][0] << points_remapped[idx][1] << points_remapped[idx][2] << "\n";
        final_points.push_back(points_remapped[idx]);
    }

    // hid_t   file_id, dataset_id, dataspace_id; /* identifiers */
    // hsize_t dims[2];
    // herr_t  status;

    // /* Create a new file using default properties. */
    // file_id = H5Fcreate(TETR_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    // /* Create the data space for the dataset. */
    // dims[0] = num_points - 1;
    // dims[1] = 3;
    // // dims[0] = 4;
    // // dims[1] = 6;
    // dataspace_id = H5Screate_simple(2, dims, NULL);

    // /* Create the dataset. */
    // dataset_id = H5Dcreate(file_id, "/data0", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // // status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, final_points.data());
    // status = H5Dclose(dataset_id);
    // status = H5Sclose(dataspace_id);
    // status = H5Fclose(file_id);
    // std::cout << status << "\n";
    // std::cout << "tetr.h5 file written to " << mesh_folder_path << "\n";
    // std::cout << "tria.h5 file written to " << mesh_folder_path << "\n";
    // std::cout << "Finished processing voxels to tetrahedral and triangle mesh!" << "\n";

    return 0;
}