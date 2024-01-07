#include <algorithm> 
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <hdf5.h>
#include <iostream>
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
#define NUM_THREADS 2

namespace po = boost::program_options;
namespace fs = boost::filesystem;

bool is_boundary_point(std::map<std::vector<int>, int> all_points, std::vector<int> check_point){
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
        if (all_points.count(neighbor_points[idx]) > 0){
            num_neighbors++;
        }
    }
    for (int idx = 0; idx < 12; idx++){
        if (all_points.count(neighbor_points_diag[idx]) > 0){
            num_neighbors_diag++;
        }
    }
    return num_neighbors != 6 || num_neighbors_diag != 12;
}

std::vector<int> get_tetrahedron(const std::vector<int>& cube_points, int tet_number){
    std::vector<int> tet;
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
            tet = {INVALID, INVALID, INVALID, INVALID};
            break;
        }
    }

    std::sort(tet.begin(), tet.end());
    return tet;
}

std::vector<std::vector<int>> get_tetrahedron_faces(const std::vector<int>& local_cube_points, const int tet_number){
    std::vector<std::vector<int>> local_tet_faces;
    switch (tet_number){
        case 0:
        {
            local_tet_faces = {
                {local_cube_points[0], local_cube_points[1], local_cube_points[4]},
                {local_cube_points[0], local_cube_points[4], local_cube_points[3]},
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
    int num_points = 1;
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

std::vector<std::vector<int>> make_cube(const std::vector<int>& coord){
    int x, y, z;
    x = coord[0];
    y = coord[1];
    z = coord[2];
    int step = 2;
    return {
            {x, y, z},
            {x + step, y, z},
            {x + step, y + step, z},
            {x, y + step, z},
            {x, y, z + step},
            {x + step, y, z + step},
            {x + step, y + step, z + step},
            {x, y + step, z + step}
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

std::vector<std::vector<int>> make_half_cubes_and_update_points(const std::vector<int> coord, std::map<std::vector<int>, int>& points, std::vector<int> voxel_shape){
    std::vector<std::vector<std::vector<int>>> half_cubes;
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
    #pragma omp critical
    add_boundary_points(half_cubes, points);
    std::vector<std::vector<int>> hcubes;
    #pragma omp critical
    for (auto& vec : half_cubes){
        std::vector<int> hcube;
        for (auto& coord : vec){
            hcube.push_back(points.at(coord));
        }
        hcubes.push_back(hcube);
    }
    return hcubes;
}

std::vector<int> remap_tetrahedrons(const std::vector<int>& tet, const std::vector<int>& remap_dict){
    return {remap_dict[tet[0]], remap_dict[tet[1]], remap_dict[tet[2]], remap_dict[tet[3]]};
}

std::vector<std::vector<int>> remap_tetrahedron_faces(const std::vector<std::vector<int>>& tet_faces, const std::vector<int>& remap_dict){
    return {
        {remap_dict[tet_faces[0][0]], remap_dict[tet_faces[0][1]], remap_dict[tet_faces[0][2]]},
        {remap_dict[tet_faces[1][0]], remap_dict[tet_faces[1][1]], remap_dict[tet_faces[1][2]]},
        {remap_dict[tet_faces[2][0]], remap_dict[tet_faces[2][1]], remap_dict[tet_faces[2][2]]},
        {remap_dict[tet_faces[3][0]], remap_dict[tet_faces[3][1]], remap_dict[tet_faces[3][2]]},
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

void invert_point(std::map<std::vector<int>, int>& points, std::map<int, std::vector<int>>& points_inverse, const std::vector<int> coord){
    if (points.contains(coord)){
        points_inverse[points.at(coord)] = coord;
    }

}

std::vector<int> make_cube_points(const std::map<std::vector<int>, int>& points, const std::vector<int> coord){
    std::vector<int> cube_points;

    std::vector<std::vector<int>> cube = make_cube(coord);
    for (int idx = 0; idx < 8; idx++){
        std::vector<int> key = {cube[idx][0], cube[idx][1], cube[idx][2]};
        if (points.contains(key)){
            cube_points.push_back(points.at(key));
        } else {
            cube_points.push_back(INVALID);
        }
    }

    return cube_points;

}

bool is_boundary_facet(std::vector<int> facet, std::map<std::vector<int>, int>& points, std::map<int, std::vector<int>>& points_inverse){
    int num_boundary = 0;
    std::cout << facet[0] << ": " << facet[1] << ": " << facet[2] << ": " << facet[3] << "\n";
    for (int& node : facet){
        std::vector<int> coord = points_inverse.at(node);
        if (is_boundary_point(points, coord)) num_boundary++;
    }

    return num_boundary == 3;
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
    omp_set_num_threads(NUM_THREADS);

    int Nx, Ny, n_points, Nz = num_files;
    std::map<std::vector<int>, int> voxels;
    std::map<std::vector<int>, int> points;

    std::vector<int> voxel_stats = {3, 3, 16};//read_input_voxels(mesh_folder_path, num_files, voxels, phase, "tif"); // {2, 2, 7};//
    voxels[{0, 0, 0}] = 1;
    voxels[{2, 0, 0}] = 1;
    voxels[{0, 2, 0}] = 1;
    voxels[{2, 2, 0}] = 0;
    voxels[{0, 0, 2}] = 1;
    voxels[{2, 0, 2}] = 1;
    voxels[{0, 2, 2}] = 1;
    voxels[{2, 2, 2}] = 1;

    Nx = voxel_stats[0];
    Ny = voxel_stats[1];
    n_points = voxel_stats[2];
    std::cout << "Read " << n_points << " coordinates from voxels of phase " << phase << "\n";

    points = build_points_from_voxels(voxels, phase, Nx, Ny, Nz);

    // build tetrahedrons and faces
    std::vector<std::vector<int>> tetrahedrons, new_tetrahedrons;
    std::vector<std::vector<std::vector<int>>> tetrahedrons_faces;
    std::vector<std::vector<int>> new_tetrahedrons_faces;

    std::cout << "Generating tetrahedrons\n";
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nx - 1; i++){
        for (int j = 0; j < Ny - 1; j++){
            for (int k = 0; k < Nz - 1; k++){
                if (!(i & 1) && !(j & 1) && !(k & 1)){
                    std::vector<int> cube_points = make_cube_points(points, {i, j, k});
                    if (std::find(cube_points.begin(), cube_points.end(), INVALID) == cube_points.end())
                    {
                        for (int idx = 0; idx < 5; idx++){
                            std::vector<int> tet = get_tetrahedron(cube_points, idx);
                            tetrahedrons.push_back(tet);
                            std::vector<std::vector<int>> tet_faces = get_tetrahedron_faces(cube_points, idx);
                            tetrahedrons_faces.push_back(tet_faces);
                        }
                    }
                    else
                    {
                        std::vector<std::vector<int>> half_cubes = make_half_cubes_and_update_points({i, j, k}, points, {Nx, Ny, Nz});
                        int n_hcubes = half_cubes.size();
                        for (int hcube_idx = 0; hcube_idx < n_hcubes; hcube_idx++){
                            std::vector<int> hcube = half_cubes[hcube_idx];
                            for (int idx = 0; idx < 5; idx++){
                                std::vector<int> tet = get_tetrahedron(hcube, idx);
                                tetrahedrons.push_back(tet);
                                std::vector<std::vector<int>> tet_faces = get_tetrahedron_faces(hcube, idx);
                                tetrahedrons_faces.push_back(tet_faces);
                            }
                        }
                    }
                }
            }
            }
        }

    std::cout << "Generated " << tetrahedrons.size() << " tetrahedrons\n";

    // points with <key,value> inverted to <value,key>
    std::map<int, std::vector<int>> points_inverse;
    std::vector<int> key;
    std::vector<int> points_id_remapping; for (int idx = 0; idx < n_points; idx++) points_id_remapping.push_back(INVALID);
    std::vector<std::vector<int>> points_remapped;

    for (int i = 0; i < Nx; i++){
        for (int j = 0; j < Ny; j++){
            for (int k = 0; k < Nz; k++){
                key = {i, j, k};
                invert_point(points, points_inverse, key);
            }
        }
    }
    std::cout << "Generated inverted points\n";

    // free up some memory
    points.clear(); voxels.clear();

    // remap points
    int num_points = 0;
    int n_tets = tetrahedrons.size();

    for (int idx = 0; idx < n_tets; idx++){
        std::vector<int> tet_points = tetrahedrons[idx];
        for (int& tet_point: tet_points){
            #pragma omp critical
            if (points_id_remapping[tet_point] == INVALID){
                points_id_remapping[tet_point] = num_points;
                std::cout << "Mapping: " << tet_point << " -> " << num_points << "\n";
                points_remapped.push_back(points_inverse[tet_point]);
                num_points ++;
            }
        }
    }

    std::cout << "Finished remapping points to account for orphaned points. " << "Total number of points: " << num_points <<"\n";

    #pragma omp parallel for
    for (int idx = 0; idx < n_tets; idx++){
        // #pragma omp critical
        {
            #pragma omp critical
            new_tetrahedrons.push_back(remap_tetrahedrons(tetrahedrons[idx], points_id_remapping));
            std::vector<std::vector<int>> local_faces = remap_tetrahedron_faces(tetrahedrons_faces[idx], points_id_remapping);
            // for (int idx = 0; idx < 4; idx++){
            //     if (is_boundary_facet(local_faces[idx], points, points_inverse)) new_tetrahedrons_faces.push_back(local_faces[idx]);
            // }
        }
    }
    for (auto& vec : new_tetrahedrons){
        std::cout << vec[0] << "," << vec[1] << "," << vec[2] << "," << vec[3] << "\n";
    }

    // free up memory
    tetrahedrons.clear(); tetrahedrons.shrink_to_fit();
    tetrahedrons_faces.clear(); tetrahedrons_faces.shrink_to_fit();

    int n_facets = new_tetrahedrons_faces.size();

    // write hdf5 file
    std::cout << "Number of of valid points = " << num_points << ", number of facets = "<< n_facets << " and number of tetrahedrons = " << n_tets << "\n";

    int total_size = 0;
    for (auto& vec : points_remapped) total_size += vec.size();

    // 2. Create a vector to hold the data.
    std::vector<int> flattened;
    flattened.reserve(total_size);

    // 3. Fill it
    for (auto& vec : points_remapped)
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
    std::vector<int> flattened_tetr;
    flattened_tetr.reserve(total_size);

    // 3. Fill it
    for (int idx1 = 0; idx1 < n_tets; idx1++){
        for (int idx2 = 0; idx2 < 4; idx2++){
            flattened_tetr.push_back(new_tetrahedrons[idx1][idx2]);
        }
    }

    // 4. Obtain the array
    auto data_tetr = flattened_tetr.data();

    dims[0] = n_tets;
    dims[1] = 4;
    dataspace_id = H5Screate_simple(2, dims, NULL);
    dataset_id = H5Dcreate(file_id, "/data1", H5T_NATIVE_INT32, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_tetr);
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

    // write triangle mesh
    dims[0] = num_points;
    dims[1] = 3;
    file_id = H5Fcreate(TRIA_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    dataspace_id = H5Screate_simple(2, dims, NULL);
    dataset_id = H5Dcreate(file_id, "/data0", H5T_NATIVE_INT32, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_0);
    H5Dclose(dataset_id);

    // triangles data
    total_size = 0;
    for (auto& vec : new_tetrahedrons_faces) total_size += vec.size();

    // 2. Create a vector to hold the data.
    std::vector<int> flattened_tria;
    flattened_tria.reserve(total_size);

    // 3. Fill it
    for (int idx1 = 0; idx1 < n_facets; idx1++){
        for (int idx2 = 0; idx2 < 3; idx2++){
            flattened_tria.push_back(new_tetrahedrons_faces[idx1][idx2]);
        }
    }

    // 4. Obtain the array
    auto data_tria = flattened_tria.data();

    dims[0] = n_facets;
    dims[1] = 3;
    dataspace_id = H5Screate_simple(2, dims, NULL);
    dataset_id = H5Dcreate(file_id, "/data1", H5T_NATIVE_INT32, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_tria);
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
    write_triangle_xdmf(num_points, n_facets);

    std::cout << "tria.h5 file written to " << mesh_folder_path << "\n";
    std::cout << "Finished processing voxels to tetrahedral and triangle mesh!" << "\n";

    return 0;
}