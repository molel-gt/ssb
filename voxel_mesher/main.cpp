#include <algorithm> 
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include "entities.h"
#include <hdf5.h>
#include <iostream>
#include <initializer_list>
#include <map>
#include <mpi.h>
#include <omp.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <ranges>
#include <vector>

#define TETR_FILE "tetr.h5"
#define TRIA_FILE "tria.h5"
#define INVALID -1
#define NUM_THREADS 8

namespace po = boost::program_options;
namespace fs = boost::filesystem;

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

void write_triangle_xdmf(int points_count, int facets_count)
{
    FILE *xdmf = 0;
    xdmf = fopen("tria.xdmf", "w");
    fprintf(xdmf, "<Xdmf Version=\"3.0\">");
    fprintf(xdmf, "<Domain>");
    fprintf(xdmf, "<Grid Name=\"Grid\">");
    fprintf(xdmf, "<Geometry GeometryType=\"XYZ\">");
    fprintf(xdmf, "<DataItem DataType=\"Float\" Dimensions=\"%d 3\" Format=\"HDF\" Precision=\"8\">tria.h5:/data0</DataItem>", points_count);
    fprintf(xdmf, "</Geometry>");
    fprintf(xdmf, "<Topology TopologyType=\"Triangle\" NumberOfElements=\"%d\" NodesPerElement=\"3\">", facets_count);
    fprintf(xdmf, "<DataItem DataType=\"Int\" Dimensions=\"%d 3\" Format=\"HDF\" Precision=\"8\">tria.h5:/data1</DataItem>", facets_count);
    fprintf(xdmf, "</Topology>");
    fprintf(xdmf, "<Attribute Name=\"name_to_read\" AttributeType=\"Scalar\" Center=\"Cell\"><DataItem DataType=\"Int\" Dimensions=\"%d\" Format=\"HDF\" Precision=\"8\">tria.h5:/data2</DataItem>", facets_count);
    fprintf(xdmf, "</Attribute>");
    fprintf(xdmf, "</Grid>");
    fprintf(xdmf, "</Domain>");
    fprintf(xdmf, "</Xdmf>");
    fclose(xdmf);
}

std::map<std::vector<int>, int> build_points_from_voxels(std::map<std::vector<int>, int> voxels, int phase, int Nx, int Ny, int Nz){
    std::map<std::vector<int>, int> output_points;
    int num_points = 1;
    for (int i = 0; i < Nx; i++){
        for (int j = 0; j < Ny; j++){
            for (int k = 0; k < Nz; k++){
                if (voxels.contains({i, j, k})){
                    if (voxels.at({i, j, k}) == phase){
                        output_points[{2 * i, 2 * j, 2 * k}] = num_points++;
                    }
                }
            }
        }
    }

    return output_points;

}

CubeType make_cube(const CoordType& coord){
    int x, y, z;
    x = coord[0];
    y = coord[1];
    z = coord[2];
    return {
            {x, y, z},
            {x + 2, y, z},
            {x + 2, y + 2, z},
            {x, y + 2, z},
            {x, y, z + 2},
            {x + 2, y, z + 2},
            {x + 2, y + 2, z + 2},
            {x, y + 2, z + 2}
        };
}

void add_boundary_points(std::vector<std::vector<std::vector<int>>>& hcubes, std::map<std::vector<int>, int>& points){
    int idx = points.size();
     for (auto& vec0 : hcubes){
        for (auto& vec1 : vec0){
            if (!points.contains(vec1)){
                points[vec1] = idx;
                idx ++;
            }
        }
     }
}

bool is_valid_half_cube(std::vector<std::vector<int>> hcube, std::vector<int> voxel_shape){
    for (auto& coord : hcube){
        if (coord[0] > voxel_shape[0] - 1 || coord[1] > voxel_shape[1] - 1 || coord[2] > voxel_shape[2] - 1) return false;
    }
    return true;
}

std::vector<CubeType> make_half_cubes_and_update_points(const std::vector<int> coord, std::map<std::vector<int>, int>& points, std::vector<int> voxel_shape){
    std::vector<CubeType> half_cubes;
    for (int slice = 0; slice < 2; slice++){
        for (int quadrant = 0; quadrant < 4; quadrant++){
            int x, y, z;
            z = coord[2] + slice;
            switch (quadrant){
                case 0:
                {
                    x = coord[0];
                    y = coord[1];
                    std::vector<std::vector<int>> hcube = {
                                {x, y, z},
                                {x + 1, y, z},
                                {x + 1, y + 1, z},
                                {x, y + 1, z},
                                {x, y, z + 1},
                                {x + 1, y, z + 1},
                                {x + 1, y + 1, z + 1},
                                {x, y + 1, z + 1}
                            };
                    if (points.contains({x, y, z + slice}) && is_valid_half_cube(hcube, voxel_shape)){
                        half_cubes.push_back(hcube);
                    }
                    break;
                }
                case 1:
                {
                    x = coord[0] + 1;
                    y = coord[1];
                    std::vector<std::vector<int>> hcube = {
                                {x, y, z},
                                {x + 1, y, z},
                                {x + 1, y + 1, z},
                                {x, y + 1, z},
                                {x, y, z + 1},
                                {x + 1, y, z + 1},
                                {x + 1, y + 1, z + 1},
                                {x, y + 1, z + 1}
                            };
                    if (points.contains({x + 1, y, z + slice}) && is_valid_half_cube(hcube, voxel_shape)){
                        half_cubes.push_back(hcube);
                    }
                    break;
                }
                case 2:
                {
                    x = coord[0] + 1;
                    y = coord[1] + 1;
                    std::vector<std::vector<int>> hcube = {
                                {x, y, z},
                                {x + 1, y, z},
                                {x + 1, y + 1, z},
                                {x, y + 1, z},
                                {x, y, z + 1},
                                {x + 1, y, z + 1},
                                {x + 1, y + 1, z + 1},
                                {x, y + 1, z + 1}
                            };
                    if (points.contains({x + 1, y + 1, z + slice})){
                        half_cubes.push_back(hcube);
                    }
                    break;
                }
                case 3:
                {
                    x = coord[0];
                    y = coord[1] + 1;
                    std::vector<std::vector<int>> hcube = {
                                {x, y, z},
                                {x + 1, y, z},
                                {x + 1, y + 1, z},
                                {x, y + 1, z},
                                {x, y, z + 1},
                                {x + 1, y, z + 1},
                                {x + 1, y + 1, z + 1},
                                {x, y + 1, z + 1}
                            };
                    if (points.contains({x, y + 1, z + slice}) && is_valid_half_cube(hcube, voxel_shape)){
                        half_cubes.push_back(hcube);
                    }
                    break;
                }
            }
        }
    }

    add_boundary_points(half_cubes, points);
    return half_cubes;
}

std::vector<int> read_input_voxels(fs::path voxels_folder, int num_files, std::map<std::vector<int>, int>& voxels, int phase, std::string ext){
    int NX, NY, n_points = 0;
    for (int idx = 0; idx < num_files; idx++){
        std::string text_idx = std::to_string(idx);
        fs::path filename = std::string(3 - text_idx.length(), '0').append(text_idx) + "." + ext;
        fs::path full_path = voxels_folder / filename;
        cv::Mat img = cv::imread(full_path.string(), cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cout << "Could not read the image.\n";
            std::vector<int> error = {-1, -1, -1};
            return error;
        }
        NX = img.cols;
        NY = img.rows;
        std::cout << "File: " << idx << ", Rows: " << NY << ", Columns: " << NX << "\n";
        for (int i = 0; i < NX; i++){
            for (int j = 0; j < NY; j++){
                cv::Vec3b value = img.at<cv::Vec3b>(j, i);
                int check_value = int(value[0]);
                if (check_value == phase){
                    voxels[{i, j, idx}] = check_value;
                    n_points++;
                }
            }
        }
    }

    std::vector<int> stats = {NX, NY, n_points};
    return stats;
}

int main(int argc, char* argv[]){
    fs::path mesh_folder_path;
    int phase, num_files;
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Creates tetrahedral and triangle mesh in xdmf format from input voxels")
    ("mesh_folder_path,MESH_FOLDER_PATH", po::value<fs::path>(&mesh_folder_path)->required(),  "mesh folder path")
    ("num_files,NUM_FILES", po::value<int>(&num_files)->required(),  "number of image files")
    ("phase,PHASE", po::value<int>(&phase)->required(),  "phase to reconstruct volume for");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    omp_set_num_threads(NUM_THREADS);

    int Nx, Ny, n_points, Nz = num_files;
    std::map<std::vector<int>, int> voxels;
    std::map<std::vector<int>, int> points;

    std::vector<int> voxel_stats = {3, 3, 30};//read_input_voxels(mesh_folder_path, num_files, voxels, phase, "tif");
    voxels[{0, 0, 0}] = 1;
    voxels[{1, 0, 0}] = 1;
    voxels[{0, 1, 0}] = 1;
    voxels[{1, 1, 0}] = 0;
    voxels[{0, 0, 1}] = 1;
    voxels[{1, 0, 1}] = 1;
    voxels[{0, 1, 1}] = 1;
    voxels[{1, 1, 1}] = 1;

    Nx = voxel_stats[0];
    Ny = voxel_stats[1];
    n_points = voxel_stats[2];
    std::cout << "Read " << n_points << " coordinates from voxels of phase " << phase << "\n";

    points = build_points_from_voxels(voxels, phase, Nx, Ny, Nz);
    Nx = 2 * Nx; Ny = 2 * Ny; Nz = 2 * Nz;

    /*
        Generate tetrahedrons and facets
    */
    std::cout << "Generating tetrahedrons\n";
    std::vector<int> tetrahedrons;
    std::vector<int> external_facets;
    std::vector<CoordType> points_mapping;
    int n_tets = 0; int n_facets = 0; n_points = 0;

    // #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nx - 1; i++){
        for (int j = 0; j < Ny - 1; j++){
            for (int k = 0; k < Nz - 1; k++){
                if (!(i & 1) && !(j & 1) && !(k & 1)){
                    CubeType cube_points = make_cube({i, j, k});
                    bool is_valid_cube = true;
                    for (int idx = 0; idx < 8; idx++) {
                        if (!points.contains(cube_points[idx])) {
                            is_valid_cube = false;
                        }
                    }
                    if (is_valid_cube)
                    {
                        std::cout << "Valid full\n";
                        for (int idx = 0; idx < 5; idx++){
                            Tetrahedron tet(cube_points, points, idx);
                            std::vector<FacetType> tet_facets = tet.get_boundary_facets();
                            std::vector<CoordType> tet_points = tet.get_points();
                            for (auto& coord : tet_points){
                                auto it = std::find(points_mapping.begin(), points_mapping.end(), coord);
                                if (it != points_mapping.end()){
                                    int pos = std::distance(points_mapping.begin(), it);
                                    tetrahedrons.push_back(pos);
                                }
                                else {
                                    points_mapping.push_back(coord);
                                    tetrahedrons.push_back(n_points);
                                    n_points++;
                                }
                            }
                            n_tets++;

                            // facets
                            for (auto& face : tet_facets){
                                for (auto& coord : face){
                                    auto it = std::find(points_mapping.begin(), points_mapping.end(), coord);
                                    if (it != points_mapping.end()){
                                        int pos = std::distance(points_mapping.begin(), it);
                                        external_facets.push_back(pos);
                                    }
                                    else {
                                        throw std::out_of_range("Invalid coord in facet");
                                    }
                                }
                                n_facets++;
                            }

                        }
                    }
                    else
                    {
                        std::cout << "Check invalids\n";
                        std::vector<CubeType> half_cubes = make_half_cubes_and_update_points({i, j, k}, points, {Nx, Ny, Nz});
                        int n_hcubes = half_cubes.size();
                        std::cout << "Number of half cubes: " << n_hcubes << "\n";
                        for (auto& hcube : half_cubes){
                            for (int idx = 0; idx < 5; idx++){
                                Tetrahedron tet(hcube, points, idx);
                                std::vector<FacetType> tet_facets = tet.get_boundary_facets();
                                std::vector<CoordType> tet_points = tet.get_points();
                                for (auto& coord : tet_points){
                                    auto it = std::find(points_mapping.begin(), points_mapping.end(), coord);
                                    if (it != points_mapping.end()){
                                        int pos = std::distance(points_mapping.begin(), it);
                                        tetrahedrons.push_back(pos);
                                    }
                                    else {
                                        points_mapping.push_back(coord);
                                        tetrahedrons.push_back(n_points);
                                        n_points++;
                                    }
                                }
                                n_tets++;

                                // facets
                                for (auto& face : tet_facets){
                                    for (auto& coord : face){
                                        auto it = std::find(points_mapping.begin(), points_mapping.end(), coord);
                                        if (it == points_mapping.end()) throw std::out_of_range("Invalid coord in facet");
                                        int pos = std::distance(points_mapping.begin(), it);
                                        external_facets.push_back(pos);
                                        std::cout << "Maliza " << pos << " has " << "\n";
                                    }
                                    n_facets++;
                                }
                            }
                        }
                    }
                }
            }
            }
        }

    std::cout << "Number of of valid points = " << n_points << ", number of facets = " << n_facets << " and number of tetrahedrons = " << n_tets << "\n";

    // write hdf5 file
    int points_size = n_points * 3 * sizeof(int);

    // 2. Create a vector to hold the data.
    std::vector<int> flattened;
    flattened.reserve(points_size);

    // 3. Fill it
    for (auto& vec : points_mapping)
        for (auto& elem : vec)
            flattened.push_back(elem);

    // 4. Obtain the array
    auto points_output = flattened.data();

    hid_t   file_id, dataset_id, dataspace_id; /* identifiers */
    hsize_t dims[2];
    // herr_t  status;

    dims[0] = n_points;
    dims[1] = 3;

    file_id = H5Fcreate(TETR_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    dataspace_id = H5Screate_simple(2, dims, NULL);
    dataset_id = H5Dcreate(file_id, "/data0", H5T_NATIVE_INT32, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, points_output);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // tetrahedrons data
    dims[0] = n_tets;
    dims[1] = 4;
    dataspace_id = H5Screate_simple(2, dims, NULL);
    dataset_id = H5Dcreate(file_id, "/data1", H5T_NATIVE_INT32, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, tetrahedrons.data());
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
    write_tetrahedral_xdmf(n_points, n_tets);
    std::cout << "tetr.h5 file written to " << mesh_folder_path << "\n";

    /* write triangle mesh */
    dims[0] = n_points;
    dims[1] = 3;
    file_id = H5Fcreate(TRIA_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    dataspace_id = H5Screate_simple(2, dims, NULL);
    dataset_id = H5Dcreate(file_id, "/data0", H5T_NATIVE_INT32, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, points_output);
    H5Dclose(dataset_id);

    dims[0] = n_facets;
    dims[1] = 3;
    dataspace_id = H5Screate_simple(2, dims, NULL);
    dataset_id = H5Dcreate(file_id, "/data1", H5T_NATIVE_INT32, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, external_facets.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Physical markers
    std::vector<int> surface_markers;
    for (int idx = 0; idx < n_facets; idx++){
        surface_markers.push_back(1);
    }
    dims[0] = n_facets;
    dims[1] = 1;
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(file_id, "/data2", H5T_NATIVE_INT32, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, surface_markers.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    H5Fclose(file_id);
    write_triangle_xdmf(n_points, n_facets);

    std::cout << "tria.h5 file written to " << mesh_folder_path << "\n";
    std::cout << "Finished processing voxels to tetrahedral and triangle mesh!" << "\n";

    return 0;
}
