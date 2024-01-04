#include <algorithm> 
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <hdf5.h>
#include <iostream>
#include <map>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
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

void write_tetrahedral_xdmf(int points_count, int tets_count)
{
    FILE *xdmf = 0;
    xdmf = fopen("tetr.xdmf", "w");
    fprintf(xdmf, "<Xdmf Version=\"3.0\">");
    fprintf(xdmf, "<Domain>");
    fprintf(xdmf, "<Grid Name=\"Grid\">");
    fprintf(xdmf, "<Geometry GeometryType=\"XYZ\">");
    fprintf(xdmf, "<DataItem DataType=\"Float\" Dimensions=\"%d 3\" Format=\"HDF\" Precision=\"8\">tetr.h5:/data0</DataItem>", points_count);
    fprintf(xdmf, "</Geometry>");
    fprintf(xdmf, "<Topology TopologyType=\"Tetrahedron\" NumberOfElements=\"%d\" NodesPerElement=\"4\">", tets_count);
    fprintf(xdmf, "<DataItem DataType=\"Int\" Dimensions=\"%d 4\" Format=\"HDF\" Precision=\"8\">tetr.h5:/data1</DataItem>", tets_count);
    fprintf(xdmf, "</Topology>");
    fprintf(xdmf, "<Attribute Name=\"name_to_read\" AttributeType=\"Scalar\" Center=\"Cell\"><DataItem DataType=\"Int\" Dimensions=\"%d\" Format=\"HDF\" Precision=\"8\">tetr.h5:/data2</DataItem>", tets_count);
    fprintf(xdmf, "</Attribute>");
    fprintf(xdmf, "</Grid>");
    fprintf(xdmf, "</Domain>");
    fprintf(xdmf, "</Xdmf>");
    fclose(xdmf);
}

int read_input_voxels(fs::path voxels_folder, int num_files, std::map<std::vector<int>, int>& voxels, std::string ext){
    for (int idx = 0; idx < num_files; idx++){
        std::string text_idx = std::to_string(idx);
        fs::path filename = std::string(3 - text_idx.length(), '0').append(text_idx) + "." + ext;
        fs::path full_path = voxels_folder / filename;
        cv::Mat3b img = cv::imread(full_path.string(), cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        if (img.empty()) {
            std::cout << "Could not read the image.\n";
            return -1;
        }
        int NX, NY;
        NX = img.size().width;
        NY = img.size().height;
        for (int i = 0; i < NX; i++){
            for (int j = 0; j < NY; j++){
                int value = img(i, j)[0];
                voxels[{i, j, idx}] = value;
            }
        }
    }
    return 0;
}

int main(int argc, char* argv[]){
    fs::path mesh_folder_path;
    int phase, num_files;
    bool boundary_layer;
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Creates tetrahedral and triangle mesh in xdmf format from input voxels")
    ("mesh_folder_path,MESH_FOLDER_PATH", po::value<fs::path>(&mesh_folder_path)->required(),  "mesh folder path")
    ("num_files,NUM_FILES", po::value<int>(&num_files)->required(),  "number of image files")
    ("phase,PHASE", po::value<int>(&phase)->required(),  "phase to reconstruct volume for")
    ("boundary_layer", po::value<bool>(&boundary_layer)->default_value(false), "whether or not to add half pixel boundary layer");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    if (!boundary_layer){
        std::cout << "No boundary layer is written\n";
    }

    int Nx = 2, Ny = 2, Nz = 2;
    std::map<std::vector<int>, int> voxels;
    std::map<std::vector<int>, int> points;

    read_input_voxels(mesh_folder_path, num_files, voxels, "tif");

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
    std::vector<std::vector<int>> tetrahedrons, new_tetrahedrons;
    std::vector<std::vector<std::vector<int>>> tetrahedrons_faces, new_tetrahedrons_faces;
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
    int n_tets = tetrahedrons.size();

    for (int idx = 0; idx < n_tets; idx++){
        std::vector<int> tet_points = tetrahedrons[idx];
        for (auto tet_point: tet_points){
            if (!points_id_remapping.contains(tet_point)){
                points_id_remapping[tet_point] = num_points;
                points_remapped[num_points] = points_inverse[tet_point];
                num_points ++;
            }
        }
    }

    for (int idx = 0; idx < n_tets; idx++){
        new_tetrahedrons.push_back(remap_tetrahedrons(tetrahedrons[idx], points_id_remapping));
        // new_tetrahedrons_faces.push_back(remap_tetrahedron_faces(tetrahedrons_faces[idx], points_id_remapping));
    }

    // write hdf5 file
    std::cout << "Number of of valid points: " << num_points << "\n";
    std::vector<std::vector<int>> final_points;

    for (int idx = 0; idx < num_points; idx++){
        final_points.push_back(points_remapped[idx]);
    }

    int total_size = 0;
    for (auto& vec : final_points) total_size += vec.size();

    // 2. Create a vector to hold the data.
    std::vector<int> flattened;
    flattened.reserve(total_size);

    // 3. Fill it
    for (auto& vec : final_points)
        for (auto& elem : vec)
            flattened.push_back(elem);

    // 4. Obtain the array
    auto data_0 = flattened.data();

    hid_t   file_id, dataset_id, dataspace_id; /* identifiers */
    hsize_t dims[2];
    // herr_t  status;

    dims[0] = num_points;
    dims[1] = 3;

    file_id = H5Fcreate(TETR_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    dataspace_id = H5Screate_simple(2, dims, NULL);
    dataset_id = H5Dcreate(file_id, "/data0", H5T_NATIVE_INT32, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_0);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // tetrahedrons data
    total_size = 0;
    for (auto& vec : new_tetrahedrons) total_size += vec.size();

    // 2. Create a vector to hold the data.
    std::vector<int> flattened_1;
    flattened_1.reserve(total_size);

    // 3. Fill it
    for (auto& vec : new_tetrahedrons)
        for (auto& elem : vec)
            flattened_1.push_back(elem);

    // 4. Obtain the array
    auto data_1 = flattened_1.data();

    dims[0] = n_tets;
    dims[1] = 4;
    dataspace_id = H5Screate_simple(2, dims, NULL);
    dataset_id = H5Dcreate(file_id, "/data1", H5T_NATIVE_INT32, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_1);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Physical markers
    std::vector<int> markers;
    for (int idx = 0; idx < n_tets; idx++){
        markers.push_back(1);
    }
    dims[0] = n_tets;
    dims[1] = 1;
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(file_id, "/data2", H5T_NATIVE_INT32, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, markers.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    H5Fclose(file_id);
    write_tetrahedral_xdmf(num_points, n_tets);
    std::cout << "tetr.h5 file written to " << mesh_folder_path << "\n";
    // std::cout << "tria.h5 file written to " << mesh_folder_path << "\n";
    std::cout << "Finished processing voxels to tetrahedral and triangle mesh!" << "\n";

    return 0;
}