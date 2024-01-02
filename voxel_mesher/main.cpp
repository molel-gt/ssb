#include <algorithm> 
#include <iostream>
#include <map>
#include <vector>
// #include <hdf5.h>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

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
            tet = {cube_points[0], cube_points[1], cube_points[3], cube_points[4]};
            break;
        }
        default:
        {
            tet = {invalid, invalid, invalid, invalid};
            break;
        }
    }

    std::sort(tet.begin(), tet.end());
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

int main(int argc, char* argv[]){
    fs::path mesh_folder_path;
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Creates tetrahedral and triangle mesh in xdmf format from input voxels")
    ("mesh_folder_path,MESH_FOLDER_PATH", po::value<fs::path>(&mesh_folder_path),  "mesh folder path")
    ("boundary_layer", po::value<bool>()->default_value(false), "whether or not to add half pixel boundary layer");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }
    
    bool boundary_layer = false;
    if (vm.count("mesh_folder_path")){
        mesh_folder_path = vm["mesh_folder_path"].as<fs::path>();
    }
    else {
        return 1;
    }
    if (vm.count("boundary_layer")){
        boundary_layer = vm["boundary_layer"].as<bool>();
    }
    std::cout << mesh_folder_path << boundary_layer << "\n";

    int Nx = 2, Ny = 2, Nz = 2;
    std::map<std::vector<int>, int> voxels;
    std::map<std::vector<int>, int> points;
    std::map<std::vector<int>, int> edges;

    voxels[{0, 0, 0}] = 1;
    voxels[{1, 0, 0}] = 1;
    voxels[{0, 1, 0}] = 1;
    voxels[{1, 1, 0}] = 0;
    voxels[{0, 0, 1}] = 1;
    voxels[{1, 0, 1}] = 1;
    voxels[{0, 1, 1}] = 1;
    voxels[{1, 1, 1}] = 1;
    // std::cout << voxels.count({13, 4, 5}) << "\n";

    // build points dictionary
    int num_points = 0;
    for (int i = 0; i < Nx; i++){
        for (int j = 0; j < Ny; j++){
            for (int k = 0; k < Nz; k++){
                if (voxels[{i, j, k}] == 1){
                    points[{i, j, k}] = num_points;
                    num_points ++;
                }
            }
        }
    }

    // build edges
    // id of points corresponding to unit cube
    int num_edges = 0;
    for (int i = 0; i < Nx; i++){
        for (int j = 0; j < Ny; j++){
            for (int k = 0; k < Nz; k++){
                int value = points[{i, j, k}];
                if (points.count({i, j, k}) > 0){
                    std::vector<std::vector<int>> neighbors = {{i + 1, j, k}, {i, j + 1}, {i, j, k + 1}, {i - 1, j, k}, {i, j - 1, k}, {i, j, k - 1}};
                    for (int idx = 0; idx < 6; idx ++){
                        int value1 = points[neighbors[idx]];
                        if (points.count(neighbors[idx]) > 0)
                        {
                            std::vector edge = {value, value1};
                            std::sort(edge.begin(), edge.end()); 
                            edges[edge] = num_edges;
                            num_edges ++;
                        }
                    }
                }
            }
        }
    }
    // build tetrahedrons and faces
    std::vector<std::vector<int>> cube;
    std::map<int, int> cube_points;
    std::vector<std::vector<int>> tetrahedrons;
    std::vector<std::vector<std::vector<int>>> tetrahedrons_faces;
    std::vector<int> tet;
    std::vector<std::vector<int>> tet_faces;
    int invalid = -1;

    for (int i = 0; i < Nx; i++){
        for (int j = 0; j < Ny; j++){
            for (int k = 0; k < Nz; k++){
                cube = {
                    {i, j, k},
                    {i + 1, j, k},
                    {i + 1, j + 1, k},
                    {i, j + 1, k},
                    {i, j, k + 1},
                    {i + 1, j, k + 1},
                    {i + 1, j + 1, k + 1},
                    {i, j + 1, k + 1}
                    };
                for (int idx = 0; idx < 8; idx++){
                    int value = points[{i, j, k}];
                    if (points.count({i, j, k}) > 0){
                        cube_points[idx] = value;
                    } else {
                        cube_points[idx] = invalid;
                    }
                }
                for (int idx = 0; idx < 5; idx++){
                    tet = get_tetrahedron(cube_points, idx);
                    if (std::find(tet.begin(), tet.end(), invalid) != tet.end()){
                        tetrahedrons.push_back(tet);
                        tet_faces = get_tetrahedron_faces(cube_points, idx);
                        tetrahedrons_faces.push_back(tet_faces);
                    }
                }
            }
        }
    }

    return 0;
}