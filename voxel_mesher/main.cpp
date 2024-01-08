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

bool is_boundary_point(const std::map<std::vector<int>, int>& all_points, std::vector<int> check_point){
    int num_neighbors = 0;
    int num_neighbors_diag = 0;
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
    std::vector<std::vector<int>> neighbor_points_diag = {
        {i, j + 1, k + 1},
        {i, j - 1, k - 1},
        {i, j - 1, k + 1},
        {i, j + 1, k - 1},

        {i + 1, j, k + 1},
        {i - 1, j, k - 1},
        {i - 1, j, k + 1},
        {i + 1, j, k - 1},

        {i + 1, j + 1, k},
        {i - 1, j - 1, k},
        {i - 1, j + 1, k},
        {i + 1, j - 1, k},
    };

    for (int idx = 0; idx < 6; idx++){
        if (all_points.contains(neighbor_points[idx])){
            num_neighbors++;
        }
    }
    for (int idx = 0; idx < 12; idx++){
        if (all_points.contains(neighbor_points_diag[idx])){
            num_neighbors_diag++;
        }
    }
    return num_neighbors != 6 || num_neighbors_diag != 12;
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
    fprintf(xdmf, "<Attribute Name=\"name_to_read\" AttributeType=\"Scalar\" Center=\"Cell\"><DataItem DataType=\"Int64\" Dimensions=\"%d\" Format=\"HDF\" Precision=\"8\">tria.h5:/data2</DataItem>", facets_count);
    fprintf(xdmf, "</Attribute>");
    fprintf(xdmf, "</Grid>");
    fprintf(xdmf, "</Domain>");
    fprintf(xdmf, "</Xdmf>");
    fclose(xdmf);
}

std::map<std::vector<int>, int> build_points_from_voxels(std::map<std::vector<int>, int> voxels, int phase, int Nx, int Ny, int Nz){
    std::map<std::vector<int>, int> output_points;
    int num_points = 0;
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

void add_point_id_if_missing(std::map<std::vector<int>, int>& points, std::vector<int> coord, int& point_id){
    if (!points.contains(coord)) points[coord] = point_id;
    point_id++;
}

std::vector<int> build_tetrahedron(std::vector<CoordType> cube, int tet_pos, const std::map<CoordType, int>& points){
    // get tetrahedrons
    std::vector<int> cube_id2point_id;
    switch(tet_pos){
        case 0:
        {
            cube_id2point_id = {0, 1, 3, 4};
            break;
        }
        case 1:
        {
            cube_id2point_id = {1, 2, 3, 6};
            break;
        }
        case 2:
        {
            cube_id2point_id = {4, 5, 6, 1};
            break;
        }
        case 3:
        {
            cube_id2point_id = {4, 7, 6, 3};
            break;
        }
        case 4:
        {
            cube_id2point_id = {4, 6, 1, 3};
            break;
        }
    }
    std::vector<int> tet_point_ids;
    for (auto& vec : cube_id2point_id){ tet_point_ids.push_back(points.at(cube[vec])); }
    return tet_point_ids;

}

std::vector<std::vector<int>> build_external_facets(std::vector<CoordType> cube, int tet_pos, const std::map<CoordType, int>& points){
    std::vector<std::vector<int>> facet_ids;
    switch (tet_pos){
        case 0:
        {
            facet_ids = {
                {0, 1, 4},
                {0, 4, 3},
                {1, 0, 3},
                {4, 1, 3}
            };
            break;
        }
        case 1:
        {
            facet_ids = {
                {1, 2, 6},
                {6, 2, 3},
                {1, 2, 3},
                {1, 6, 3}
            };
            break;
        }
        case 2:
        {
            facet_ids = {
                {1, 5, 4},
                {1, 6, 5},
                {4, 5, 6},
                {4, 6, 1}
            };
            break;
        }
        case 3:
        {
            facet_ids = {
                {4, 7, 3},
                {3, 7, 6},
                {4, 7, 6},
                {4, 6, 3}
            };
            break;
        }
        case 4:
        {
            facet_ids = {
                {4, 3, 1},
                {1, 3, 6},
                {4, 1, 6},
                {4, 5, 6}
            };
            break;
        }
    }
    std::vector<std::vector<int>> external_facets;
    for (auto& facet : facet_ids){
        std::vector<int> lfacet;
        int num_on_boundary = 0;
        for (auto& vec : facet){
            lfacet.push_back(points.at(cube[vec]));
            if (is_boundary_point(points, cube[vec])) num_on_boundary++;
        }
        if (num_on_boundary == 3) external_facets.push_back(lfacet);
    }
    return external_facets;
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

    std::vector<int> voxel_stats = read_input_voxels(mesh_folder_path, num_files, voxels, phase, "tif");
    // std::vector<int> voxel_stats = {2, 2, 30};
    // voxels[{0, 0, 0}] = 1;
    // voxels[{1, 0, 0}] = 1;
    // voxels[{0, 1, 0}] = 1;
    // voxels[{1, 1, 0}] = 0;
    // voxels[{0, 0, 1}] = 1;
    // voxels[{1, 0, 1}] = 1;
    // voxels[{0, 1, 1}] = 1;
    // voxels[{1, 1, 1}] = 1;

    Nx = voxel_stats[0];
    Ny = voxel_stats[1];
    n_points = voxel_stats[2];
    std::cout << "Read " << n_points << " coordinates from voxels of phase " << phase << "\n";

    points = build_points_from_voxels(voxels, phase, Nx, Ny, Nz);

    /*
        Generate tetrahedrons and facets
    */
    std::cout << "Generating tetrahedrons\n";
    std::vector<std::vector<int>> tetrahedrons;
    std::vector<std::vector<int>> tetrahedrons_ids;
    std::vector<std::vector<int>> external_facets_ids;
    std::vector<CoordType> points_mapping;
    int n_tets = 0;
    int64_t n_facets = 0;

    int point_id = n_points;

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nx - 1; i++){
        for (int j = 0; j < Ny - 1; j++){
            for (int k = 0; k < Nz - 1; k++){
                CubeType cube = make_cube({2 * i, 2 * j, 2 * k});
                bool is_valid_cube = true;
                for (int idx = 0; idx < 8; idx++) {
                    if (!points.contains(cube[idx])) { is_valid_cube = false; }
                }
                if (is_valid_cube)
                {
                    #pragma omp critical
                    for (int idx = 0; idx < 5; idx++){
                        std::vector<int> tet = build_tetrahedron(cube, idx, points);
                        tetrahedrons_ids.push_back(tet); n_tets++;
                        // std::vector<std::vector<int>> efacet_ids = build_external_facets(cube, idx, points);
                        // for (auto& fct_id : efacet_ids) { external_facets_ids.push_back(fct_id); n_facets++; }
                    }
                }
                else
                {
                    std::vector<CubeType> half_cubes = make_half_cubes_and_update_points({2 * i, 2 * j, 2 * k}, points, {2 * Nx, 2 * Ny, 2 * Nz});
                    for (auto& cube : half_cubes){
                        for (auto& coord : cube) {
                            add_point_id_if_missing(points, coord, point_id);
                        }
                    }
                    for (auto& cube : half_cubes) {
                        #pragma omp critical
                        for (int idx = 0; idx < 5; idx++){
                            std::vector<int> tet = build_tetrahedron(cube, idx, points);
                            std::sort(tet.begin(), tet.end());
                            tetrahedrons_ids.push_back(tet); n_tets++;
                            // std::vector<std::vector<int>> lfacet_ids = build_external_facets(cube, idx, points);
                            // for (auto& fct_id : lfacet_ids) { external_facets_ids.push_back(fct_id); n_facets++; }
                        }
                    }
                }
        }
            }
        }

    std::cout << "Number of of valid points = " << point_id << ", number of facets = " << n_facets << " and number of tetrahedrons = " << n_tets << "\n";

    int counter = 0;
    std::vector<int> ids_remapping;
    ids_remapping.reserve(point_id * sizeof(int));
    for (int idx = 0; idx < point_id; idx++) {
        #pragma omp critical
        ids_remapping.push_back(INVALID);
    }

    std::map<int, CoordType> points_inverse;
    for (auto& kv : points){
        #pragma omp critical
        points_inverse[kv.second] = kv.first;
    }

    std::cout << "Finished generating ids for remapping\n";

    for (auto& tet : tetrahedrons_ids){
        for (auto& coord_id : tet){
            if (ids_remapping[coord_id] == INVALID) {
                #pragma omp critical
                {
                    ids_remapping[coord_id] = counter;
                    counter++;
                }
                CoordType coord = points_inverse[coord_id];
                #pragma omp critical
                points_mapping.push_back(coord);
            }
        }
    }

    std::cout << "Finished remapping tetrahedrons\n";
    std::vector<int> tets_flat; tets_flat.reserve(n_tets * 4 * sizeof(int));
    // std::vector<int> efacets_flat; efacets_flat.reserve(n_facets * 4 * sizeof(int));
    std::vector<int> flattened; flattened.reserve(counter * 3 * sizeof(int));

    // write hdf5 file

    // Coordinates
    for (auto& vec : points_mapping)
        #pragma omp critical
        for (auto& elem : vec) flattened.push_back(elem);
    auto points_output = flattened.data();

    hid_t   file_id, dataset_id, dataspace_id; /* identifiers */
    hsize_t dims[2];
    // herr_t  status;

    dims[0] = counter;
    dims[1] = 3;

    file_id = H5Fcreate(TETR_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    dataspace_id = H5Screate_simple(2, dims, NULL);
    dataset_id = H5Dcreate(file_id, "/data0", H5T_NATIVE_INT32, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, points_output);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // tetrahedrons data
    int tets_size = n_tets * 4 * sizeof(int);

    // Create a vector to hold the data.
    std::vector<int> flattened_tets; flattened_tets.reserve(tets_size);

    for (auto& vec : tetrahedrons_ids)
        for (auto& elem : vec) flattened_tets.push_back(ids_remapping[elem]);

    dims[0] = n_tets;
    dims[1] = 4;
    dataspace_id = H5Screate_simple(2, dims, NULL);
    dataset_id = H5Dcreate(file_id, "/data1", H5T_NATIVE_INT32, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, flattened_tets.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Physical markers
    std::vector<int> markers;
    for (int idx = 0; idx < n_tets; idx++) markers.push_back(1);

    dims[0] = n_tets;
    dims[1] = 1;
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(file_id, "/data2", H5T_NATIVE_INT32, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, markers.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    H5Fclose(file_id);
    write_tetrahedral_xdmf(counter, n_tets);
    std::cout << "tetr.h5 file written to " << mesh_folder_path << "\n";

    // /* write triangle mesh */
    // dims[0] = counter;
    // dims[1] = 3;
    // file_id = H5Fcreate(TRIA_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    // dataspace_id = H5Screate_simple(2, dims, NULL);
    // dataset_id = H5Dcreate(file_id, "/data0", H5T_NATIVE_INT32, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, points_output);
    // H5Dclose(dataset_id);

    // std::vector<int> flattened_facets; flattened_facets.reserve(n_facets * 3 * sizeof(int));
    // for (auto& vec : external_facets_ids)
    //     for (auto& elem : vec) flattened_facets.push_back(ids_remapping[elem]);

    // dims[0] = n_facets;
    // dims[1] = 3;
    // dataspace_id = H5Screate_simple(2, dims, NULL);
    // dataset_id = H5Dcreate(file_id, "/data1", H5T_NATIVE_INT32, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, flattened_facets.data());
    // H5Dclose(dataset_id);
    // H5Sclose(dataspace_id);

    // // Physical markers
    // std::vector<int64_t> surface_markers;
    // for (int idx = 0; idx < n_facets; idx++) surface_markers.push_back(1);

    // dims[0] = n_facets;
    // dims[1] = 1;
    // dataspace_id = H5Screate_simple(1, dims, NULL);
    // dataset_id = H5Dcreate(file_id, "/data2", H5T_NATIVE_INT64, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // H5Dwrite(dataset_id, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, surface_markers.data());
    // H5Dclose(dataset_id);
    // H5Sclose(dataspace_id);

    // H5Fclose(file_id);
    // write_triangle_xdmf(n_points, n_facets);

    std::cout << "tria.h5 file written to " << mesh_folder_path << "\n";
    std::cout << "Finished processing voxels to tetrahedral and triangle mesh!" << "\n";

    return 0;
}
