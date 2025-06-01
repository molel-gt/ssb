#include "merge_tetgen_outputs.h"


int main(int argc, char** argv){
    std::filesystem::path input_dir = "output/segmentation/201-201-201/0-0-0/";
    std::vector<std::filesystem::path> input_nodes_files = {input_dir / "voids.1.node", input_dir / "sse.1.node", input_dir / "cam.1.node"};
    std::vector<std::filesystem::path> input_faces_files = {input_dir / "voids.1.face", input_dir / "sse.1.face", input_dir / "cam.1.face"};
    std::vector<std::filesystem::path> input_tets_files = {input_dir / "voids.1.ele", input_dir / "sse.1.ele", input_dir / "cam.1.ele"};

    std::filesystem::path output_nodes_file = input_dir / "tomo.1.node";
    std::filesystem::path output_faces_file = input_dir / "tomo.1.face";
    std::filesystem::path output_tets_file = input_dir / "tomo.1.ele";

    std::map<std::string, std::map<int, int>> nodes_lookup;
    merge_tetgen_nodes(input_nodes_files, output_nodes_file, nodes_lookup);
    merge_tetgen_tets(input_tets_files, output_tets_file, nodes_lookup);
    return 0;
}

// void strip_leading_hash_character(std::filesystem::path input_file){
//     const char* sed_cmd = strcat("sed -i \'/#/d\' ",input_file.c_str());
//     std::system(sed_cmd);
// }

std::vector<float> split_string_into_array(char* text, const char* delimiter){
    char *token = strtok(text, delimiter);
    std::vector<float> output;
    int count = 0;
    while (token != NULL)
    {
        output.push_back(std::stof(token));
        count ++;
        token = strtok(NULL, delimiter);
    }

    return output;
    
}

std::map<Point, std::vector<int>> read_tetgen_nodes_to_map(std::filesystem::path nodes_file){
    std::map<Point, std::vector<int>> output_nodes;
    std::ifstream file(nodes_file);
    std::string line;
    int count = 0;
    int n_entities;
    int entity_size = 3;
    bool attribute = false;
    bool boundary = false; 
    if (file.is_open()) {
        while (getline(file, line)) {
            int idx = 0;
            char* line_text;
            strcpy(line_text, line.c_str());
            std::vector<float> parts = split_string_into_array(line_text, " ");
            if (count == 0){
                n_entities = parts[0];
                entity_size = parts[1];
                attribute = parts[2];
                boundary = parts[3];
            }
            else {
                int idx = parts[0];
                int attribute_val = -1;
                int boundary_val = -1;
                Point p = {parts[1], parts[2], parts[3]};
                if (parts.size() == 5){
                    if (attribute){ attribute_val = parts[4]; }
                    if (boundary){ boundary_val = parts[4]; }
                }
                else if (parts.size() == 6){
                    attribute_val = parts[4];
                    boundary_val = parts[4];
                }
            }
            count ++;
        }
        file.close();
    }
    else {
        std::cerr << "Unable to open file!" << std::endl;
    }

    std::cout <<  "Read " << n_entities << " coordinates from file " << nodes_file << "\n";

    return output_nodes;

}

void merge_tetgen_nodes(std::vector<std::filesystem::path> input_nodes_files, std::filesystem::path output_nodes_file, std::map<std::string, std::map<int, int>>& nodes_lookup){
    std::map<Point, std::vector<int>> merged_nodes;
    int file_count = 0;
    int node_idx = 0;
    for (const auto& nodes_file : input_nodes_files){
        std::map<Point, std::vector<int>> nodes = read_tetgen_nodes_to_map(nodes_file);
        std::string lookup_key = nodes_file.filename().string().substr(0, 3);
        if (file_count == 0) {
            merged_nodes.insert(nodes.begin(), nodes.end());
            for (const auto& pair : nodes){
                nodes_lookup[lookup_key][pair.second[0]] = pair.second[0];
                node_idx = pair.second[0];
            }
        }
        else {
            for (const auto& pair : nodes){
                if (merged_nodes.contains(pair.first)){
                    std::vector<int> value = merged_nodes[pair.first];
                    nodes_lookup[lookup_key][pair.second[0]] = value[0];
                }
                else {
                    node_idx ++;
                    if (pair.second.size() == 1){
                        merged_nodes[pair.first] = {node_idx};
                    }
                    else if (pair.second.size() == 2){
                        merged_nodes[pair.first] = {node_idx, pair.second[1]};
                    }
                    else if (pair.second.size() == 3){
                        merged_nodes[pair.first] = {node_idx, pair.second[1], pair.second[2]};
                    }
                    nodes_lookup[lookup_key][pair.second[0]] = node_idx;
                }
            }
        }
        file_count ++;
    }

    write_nodes_to_file(merged_nodes, output_nodes_file);

}

void renumber_tetgen_nodes(std::map<int, int> old_to_new){

}

void renumber_tetgen_faces(std::map<int, int> old_to_new);

void renumber_tetgen_tets(std::map<int, int> old_to_new);

std::map<Triangle, std::vector<int> > read_tetgen_faces_to_map(std::filesystem::path faces_file){
    std::map<Triangle, std::vector<int>> output_faces;
    std::ifstream file(faces_file);
    std::string line;
    int count = 0;
    int n_entities;
    int entity_size = 3;
    bool attribute = false;
    bool boundary = false; 
    if (file.is_open()) {
        while (getline(file, line)) {
            char* line_text;
            strcpy(line_text, line.c_str());
            std::vector<float> parts = split_string_into_array(line_text, " ");
            int boundary_val = -1;
            if (count == 0){
                n_entities = parts[0];
                entity_size = parts[1];
                boundary = parts[2];
            }
            else {
                int idx = parts[0];
                Triangle f = {parts[1], parts[2], parts[3]};
                if (parts.size() == 5 && boundary){ boundary_val = parts[4]; }
                output_faces[f] = {idx, boundary_val};
            }
            
            count ++;
        }
        file.close();
    }
    else {
        std::cerr << "Unable to open file!" << std::endl;
    }
    std::cout <<  "Read " << n_entities << " triangles from file " << faces_file << "\n";

    return output_faces;
}

void merge_tetgen_faces(std::vector<std::filesystem::path> node_files, std::string faces_files){

}

std::map<Tetrahedron, std::vector<int> > read_tetgen_tets_to_map(std::filesystem::path tets_file){
    std::map<Tetrahedron, std::vector<int>> output_tets;
    std::ifstream file(tets_file);
    std::string line;
    int count = 0;
    int n_entities;
    int entity_size = 4;
    bool boundary = false; 
    if (file.is_open()) {
        while (getline(file, line)) {
            char* line_text;
            strcpy(line_text, line.c_str());
            std::vector<float> parts = split_string_into_array(line_text, " ");
            if (count == 0){
                n_entities = parts[0];
                entity_size = parts[1];
                boundary = parts[2];
            }
            else {
                int idx = parts[0];
                int attribute_val = -1;
                int boundary_val = -1;
                Tetrahedron t = {parts[1], parts[2], parts[3], parts[4]};
                if (parts.size() == 6){
                    if (boundary){
                        boundary_val = parts[5];
                    }
            }
            output_tets[t] = {idx, boundary_val};
            count ++;
        }
        file.close();
    }
}
    else {
        std::cerr << "Unable to open file!" << std::endl;
    }
    std::cout <<  "Read " << n_entities << " tetrahedrons from file " << tets_file << "\n";

    return output_tets;

}

void merge_tetgen_tets(std::vector<std::filesystem::path> tets_files, std::filesystem::path output_tets_file, std::map<std::string, std::map<int, int>>& nodes_lookup){
    std::map<Tetrahedron, std::vector<int>> tets;
    int file_count = 0;
    int tets_count = 0;
    for (auto& tets_file : tets_files){
        std::map<Tetrahedron, std::vector<int>> raw_tets = read_tetgen_tets_to_map(tets_file);
        std::string lookup_key = tets_file.filename().string().substr(0, 3);
        std::map<int, int> lookup_table = nodes_lookup[lookup_key];
        if (file_count == 0){
            tets.insert(raw_tets.begin(), raw_tets.end());
            tets_count += raw_tets.size();
        }
        else {
            for (auto& pair : raw_tets){
                Tetrahedron new_key = {lookup_table.at(pair.first[0]), lookup_table.at(pair.first[1]), lookup_table.at(pair.first[2]), lookup_table.at(pair.first[3])};
                tets[new_key] = {pair.second[0] + tets_count, pair.second[1] + tets_count};
            }
            tets_count += raw_tets.size();

        }
        file_count += 1;

    }
    write_tets_to_file(tets, output_tets_file);
}

void write_nodes_to_file(const std::map<Point, std::vector<int>>& nodes, std::filesystem::path output_nodes_file){
     std::ofstream outputFile(output_nodes_file);
     outputFile << nodes.size() << " " << 3 << " " << 0 << " " << 0 << "\n";
     if (outputFile.is_open()) {
        for (const auto& pair : nodes) {
            outputFile << pair.second[0] << " " << pair.first[0] << " " << pair.first[1] << " " << pair.first[2] << "\n";
        }
        outputFile.close();
        std::cout << "Data successfully written to " << output_nodes_file << std::endl;
    } else {
        std::cerr << "Error: Unable to open the file for writing." << std::endl;
    }
}

void write_faces_to_file(std::map<Triangle, std::vector<int> >& faces, std::filesystem::path output_faces_file){
    std::ofstream outputFile(output_faces_file);
     outputFile << faces.size() << " " << 3 << " " << 0 << " " << 0 << "\n";
     if (outputFile.is_open()) {
        for (const auto& pair : faces) {
            if (pair.second.size() == 2){
                outputFile << pair.second[0] << " " << pair.first[0] << " " << pair.first[1] << " " << pair.first[2] << "\n";
            }
            else {
                outputFile << pair.second[0] << " " << pair.first[0] << " " << pair.first[1] << "\n";
            }
        }
        outputFile.close();
        std::cout << "Data successfully written to " << output_faces_file << std::endl;
    } else {
        std::cerr << "Error: Unable to open the file for writing." << std::endl;
    }
}

void write_tets_to_file(const std::map<Tetrahedron, std::vector<int>>& tets, std::filesystem::path output_tets_file){
    std::ofstream outputFile(output_tets_file);
     outputFile << tets.size() << " " << 4 << " " << 0 << " " << 0 << "\n";
     if (outputFile.is_open()) {
        for (const auto& pair : tets) {
            if (pair.second.size() == 0){
                outputFile << pair.second[0] << " " << pair.first[0] << " " << pair.first[1] << " " << pair.first[2] << " " << pair.first[3] << "\n";
            }
            else {
                outputFile << pair.second[0] << " " << pair.first[0] << " " << pair.first[1] << " " << pair.first[2] << " " << pair.first[3] << " " << pair.second[1] << "\n";
            }
        }
        outputFile.close();
        std::cout << "Data successfully written to " << output_tets_file << std::endl;
    } else {
        std::cerr << "Error: Unable to open the file for writing." << std::endl;
    }
}
