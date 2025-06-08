#include "voxels_to_tets.h"


int main(int argc, char** argv){
    std::filesystem::path input_dir;
    int h_max;
    po::options_description desc("Options");
    desc.add_options()
        ("input_dir,i", po::value<std::filesystem::path>(&input_dir)->required(), "directory containing input files")
        ("h_max,h", po::value<int>(&h_max)->required(), "maximum unscaled mesh size");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    std::vector<std::filesystem::path> input_files = {input_dir / "voids.dat", input_dir / "cam.dat", input_dir / "sse.dat"};
    std::vector<std::filesystem::path> output_tetgen_node_files = {input_dir / "voids.node", input_dir / "cam.node", input_dir / "sse.node"};
    std::vector<std::filesystem::path> output_tetgen_ele_files = {input_dir / "voids.ele", input_dir / "cam.ele", input_dir / "sse.ele"};

    for (int file_id = 0; file_id < 3; file_id++){
        std::filesystem::path dat_file = input_files[file_id];
        std::cout << "Processing " << dat_file << std::endl;
        std::map<Coordinate, int> points;
        std::cout << "Reading phase data..\n";
        read_phase_data(dat_file, points);
        std::vector<std::array<int, 8>> cubes;
        int idx = 0;
        const int N_points = points.size();
        std::vector<Coordinate> all_coords;
        for (auto& pair : points){
            int idx = pair.second;
            Coordinate coord = (std::array<int, 3>)pair.first;
            all_coords.push_back(coord);
        }
        std::cout << "Number of points " << points.size() << std::endl;
        for (int point_id=0; point_id < N_points; point_id++){
            Coordinate coord = all_coords.at(point_id);
            int x = coord[0];
            int y = coord[1];
            int z = coord[2];
            // int h_max = 5;
            if (x % h_max == 0 && y % h_max == 0 && z % h_max == 0){
                std::array<Coordinate, 8> in_cube = make_cube(points, coord, h_max);
                if (!in_cube.empty()){
                    try {
                        std::array<int, 8> cids = cube_coords_to_cube_ids(points, in_cube);
                        std::set cids_set(cids.begin(), cids.end());
                        if (cids_set.size() == 8) cubes.push_back(cids);
                        idx ++;
                    }
                    catch (std::out_of_range){};
                }
                else {
                    for (int i=0; i < h_max; i++){
                        for (int j=0; j < h_max; j++){
                            for (int k=0; k < h_max; k++){
                                Coordinate coord = {x+i, y+j, z+k};
                                std::array<Coordinate, 8> small_cube = make_cube(points, coord, 1);
                                if (!small_cube.empty()){
                                    try {
                                        std::array<int, 8> cids = cube_coords_to_cube_ids(points, in_cube);
                                        std::set cids_set(cids.begin(), cids.end());
                                        if (cids_set.size() == 8) cubes.push_back(cids);
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
        std::vector<Tetrahedron> tets;
        int tets_counter = 0;
        for (auto& cube : cubes){
            // std::cout << cube[0] << std::endl;
            if (!cube.empty()){
                std::array<Tetrahedron, 5> cube_tets = make_tetrahedrons_from_cube(cube);
                for (int i=0; i < 5; i++){
                    Tetrahedron tet = cube_tets[i];
                    tets.push_back(tet);
                }
            }
        }
        // write nodes to file
        std::filesystem::path output_nodes_file = output_tetgen_node_files[file_id];
        write_tetgen_node_file(output_nodes_file, points);
        // // write tets to file
        std::filesystem::path output_tets_file = output_tetgen_ele_files[file_id];
        write_tetgen_ele_file(output_tets_file, tets, tets_counter);
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
            idx ++;
            points[coord] = idx;
        }
        file.close();
    }
    else {
        std::cerr << "Unable to open file!" << std::endl;
    }
}

std::array<Coordinate, 8> make_cube(const std::map<Coordinate, int>& points, Coordinate& coord, int h){
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


bool cube_is_filled(const std::map<Coordinate, int>& points, Coordinate& coord, int h){
    int x, y, z;
    x = coord[0]; y = coord[1]; z = coord[2];
    int counter = 0;
    for (int i = 0; i < h+1; i++){
        for (int j = 0; j < h+1; j++){
            for (int k = 0; k < h+1; k++){
                Coordinate c = {x + i, y + j, z + k};
                if (points.count(c) > 0) counter ++;
            }
        }
    }
    return counter == (h + 1) * (h + 1) * (h + 1);
}

std::array<int, 8> cube_coords_to_cube_ids(const std::map<Coordinate, int>& points, std::array<Coordinate, 8>& in_cube){
    std::array<int, 8> out_cube;
    for (int idx = 0; idx < 8; idx++){
        Coordinate coord = in_cube[idx];
        int point_id = points.at(coord);
        if (point_id > 0) out_cube[idx] = point_id;
    }
    // std::cout << "******************************************\n";
    // printf("%d,%d\n", out_cube[0], out_cube[7]);
    // std::cout << "******************************************\n";

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

void write_tetgen_ele_file(std::filesystem::path output_tets_file, const std::vector<Tetrahedron>& tets, int num_tets){
    std::ofstream outputFile(output_tets_file);
     outputFile << tets.size() << " " << 4 << " " << 0 << " " << 0 << "\n";
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
