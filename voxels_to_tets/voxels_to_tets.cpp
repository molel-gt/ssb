#include "voxels_to_tets.h"


int main(int argc, char** argv){
    std::filesystem::path input_dir;
    po::options_description desc("Options");
    desc.add_options()
        ("input_dir,i", po::value<std::filesystem::path>(&input_dir)->required(), "directory containing input files");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    omp_set_num_threads(4);

    std::vector<std::filesystem::path> input_files = {input_dir / "voids.dat", input_dir / "cam.dat", input_dir / "sse.dat"};
    std::vector<std::filesystem::path> output_tetgen_node_files = {input_dir / "voids.node", input_dir / "cam.node", input_dir / "sse.node"};
    std::vector<std::filesystem::path> output_tetgen_ele_files = {input_dir / "voids.ele", input_dir / "cam.ele", input_dir / "sse.ele"};

    for (int file_id = 0; file_id < 3; file_id++){
        std::filesystem::path dat_file = input_files[file_id];
        std::cout << "Processing " << dat_file << std::endl;
        std::map<Coordinate, int> points;
        std::cout << "Reading phase data..\n";
        read_phase_data(dat_file, points);
        std::array<std::array<int, 8>, N> cubes;
        int idx = 0;
        auto ks = std::views::values(points);
        std::vector<int> values{ ks.begin(), ks.end() };
        auto max_element_it = std::max_element(values.begin(), values.end());
        const int N_points = points.size();
        std::vector<Coordinate> all_coords;
        for (auto& pair : points){
            int idx = pair.second;
            Coordinate coord = (std::array<int, 3>)pair.first;
            all_coords.push_back(coord);
        }
        std::cout << "Number of points " << points.size() << std::endl;
        // #pragma omp for
        // #pragma omp for
        for (int point_id=0; point_id < N_points; point_id++){
            Coordinate coord = all_coords.at(point_id);
            int x = coord[0];
            int y = coord[1];
            int z = coord[2];
            if (x % 2 == 0 && y % 2 == 0 && z % 2 == 1){
                std::array<Coordinate, 8> in_cube = make_cube(points, coord, 2);
                if (!in_cube.empty()){
                    try {
                        cubes[idx] = cube_coords_to_cube_ids(points, in_cube);
                        idx ++;
                    }
                    catch (std::out_of_range){};
                }
                else {
                    #pragma omp parallel for num_threads(4)
                    for (int i=0; i < 2; i++){
                        for (int j=0; j < 2; j++){
                            for (int k=0; k < 2; k++){
                                std::cout << "Running program with " << omp_get_thread_num() << " threads\n";
                                Coordinate coord = {x+i, y+j, z+k};
                                std::array<Coordinate, 8> small_cube = make_cube(points, coord, 1);
                                if (!small_cube.empty()){
                                    try {
                                        cubes[idx] = cube_coords_to_cube_ids(points, small_cube);
                                        idx ++;
                                    }
                                    catch (std::out_of_range){};
                                    
                                }
                            }
                        }
                    }
                }
            }
        }
        // Process tetrahedrons
        std::array<Tetrahedron, N_tets> tets;
        int tets_counter = 0;
        // #pragma omp for
        for (int idx=0; idx < N; idx++){
            std::array<int, 8> cube = cubes[idx];
            if (!cube.empty()){
                std::array<Tetrahedron, 5> cube_tets = make_tetrahedrons_from_cube(cube);
                for (int i=0; i < 5; i++){
                    Tetrahedron tet = cube_tets[i];
                    tets[tets_counter] = tet;
                    tets_counter ++;
                }
            }
        }
        // write nodes to file
        std::filesystem::path output_nodes_file = output_tetgen_node_files[file_id];
        write_tetgen_node_file(output_nodes_file, points);
        // // write tets to file
        std::filesystem::path output_tets_file = output_tetgen_ele_files[file_id];
        // std::cout << tets[0][0] << " "<< tets[0][1] << " " << tets[0][2] << " " << tets[0][0] << std::endl;
        // write_tetgen_ele_file(output_tets_file, tets, tets_counter);
    }

    return 0;
}


void split_string_into_array(char* text, const char* delimiter, std::vector<int>& output){
    char *token = strtok(text, delimiter);
    while (token != NULL)
    {
        try {
            int num = std::stoi(token);
            output.push_back(num);
            token = strtok(NULL, delimiter);
        }
        catch (std::invalid_argument) {
        std::cout << text << std::endl;
    }    
    }
    
}

void read_phase_data(std::filesystem::path input_file, std::map<Coordinate, int>& points){
    std::ifstream file(input_file);
    std::string line;
    int idx = 0;
    if (file.is_open()) {
        while (getline(file, line)) {
            char line_text[100];
            const char* old_line = line.c_str();
            strcpy(line_text, old_line);
            std::vector<int> parts;
            split_string_into_array(line_text, ",", parts);
            Coordinate coord = {parts[0], parts[1], parts[2]};
            points[coord] = idx;
            idx ++;
        }
        file.close();
    }
    else {
        std::cerr << "Unable to open file!" << std::endl;
    }
}

std::array<Coordinate, 8> make_cube(std::map<Coordinate, int>& points, Coordinate& coord, int h){
    int x = coord[0]; int y = coord[1]; int z = coord[2];
    std::array<Coordinate, 8> cube_coords;
    cube_coords[0] = {x, y, z};
    cube_coords[1] = {x+h, y, z};
    cube_coords[2] = {x+h, y+h, z};
    cube_coords[3] = {x, y+h, z};
    cube_coords[4] = {x, y, z+h};
    cube_coords[5] = {x+h, y, z+h};
    cube_coords[6] = {x+h, y+h, z+h};
    cube_coords[7] = {x, y+h, z+h};
    std::array<Coordinate, 8> empty_cube;
    if (cube_is_filled(points, coord, h)) { return cube_coords; } else { return empty_cube; }
}


bool cube_is_filled(std::map<Coordinate, int>& points, Coordinate& coord, int h){
    int x, y, z;
    x = coord[0]; y = coord[1]; z = coord[2];
    int counter = 0;
    for (int i = 0; i < h+1; i++){
        for (int j = 0; j < h+1; j++){
            for (int k = 0; k < h+1; k++){
                Coordinate c = {x + i, y + j, z + k};
                if (points.count(c)) counter ++;
            }
        }
    }
    return counter == (h + 1) * (h + 1) * (h + 1);
}

std::array<int, 8> cube_coords_to_cube_ids(const std::map<Coordinate, int>& points, std::array<Coordinate, 8>& in_cube){
    std::array<int, 8> out_cube;
    for (int idx = 0; idx < 8; idx++){
        Coordinate coord = in_cube[idx];
        // try {
        out_cube[idx] = points.at(coord);
    // }
    // catch (std::out_of_range){
    //     std::cout << coord[0] << "," << coord[1] << "," << coord[2] << std::endl;
    //     throw;
    // }
    }

    return out_cube;
}

std::array<Tetrahedron, 5> make_tetrahedrons_from_cube(const std::array<int, 8>& cube){
    std::array<Tetrahedron, 5> cube_tets;
    cube_tets[0] = {cube[0], cube[1], cube[3], cube[4]};
    cube_tets[1] = {cube[1], cube[2], cube[3], cube[6]};
    cube_tets[2] = {cube[4], cube[5], cube[6], cube[1]};
    cube_tets[3] = {cube[4], cube[7], cube[6], cube[3]};
    cube_tets[4] = {cube[4], cube[6], cube[1], cube[3]};

    return cube_tets;
}

void write_tetgen_node_file(std::filesystem::path output_nodes_file, const std::map<Coordinate, int>& nodes){
    std::ofstream outputFile(output_nodes_file);
     outputFile << nodes.size() << " " << 3 << " " << 0 << " " << 0 << "\n";
     if (outputFile.is_open()) {
        for (const auto& pair : nodes) {
            outputFile << pair.second << " " << pair.first[0] << " " << pair.first[1] << " " << pair.first[2] << "\n";
        }
        outputFile.close();
        std::cout << "Data successfully written to " << output_nodes_file << std::endl;
    } else {
        std::cerr << "Error: Unable to open the file for writing." << std::endl;
    }
}

void write_tetgen_ele_file(std::filesystem::path output_tets_file, const std::array<Tetrahedron, N_tets>& tets, int num_tets){
    std::ofstream outputFile(output_tets_file);
     outputFile << num_tets << " " << 4 << " " << 0 << " " << 0 << "\n";
     int idx = 0;
     if (outputFile.is_open()) {
        for (const auto& tet : tets) {
            idx ++;
            outputFile << idx << " " << tet[0] << " " << tet[1] << " " << tet[2] << " " << tet[3] << "\n";
        }
        outputFile.close();
        std::cout << "Data successfully written to " << output_tets_file << std::endl;
    } else {
        std::cerr << "Error: Unable to open the file for writing." << std::endl;
    }
}
