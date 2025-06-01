#include "merge_tetgen_outputs.h"


int main(int argc, char** argv){
    return 0;
}

void strip_leading_character(std::string input_file, char character){
    std::system("sed -i '/#/d'" + " " + std::format("{}", input_file));
}

template <typename std::vector<T>> split_string_into_array(char[] text, char delimiter){
    char *token = strtok(text, delimiter);
    std::vector<T> output;
    int count = 0;
    while (token != NULL)
    {
        output.push_back(T()token);
        count ++;
        token = strtok(NULL, delimiter);
    }

    return output;
    
}

std::map<Point, std::vector<int>> read_tetgen_nodes_to_map(std::string nodes_file){
    // remove lines begining with #
    std::map<Point, std::vector<int>> output_nodes;
    strip_leading_character(nodes_file, "#");
    ifstream file(nodes_file);
    std::string line;
    int count = 0;
    int n_entities;
    int entity_size = 3;
    bool attribute = false;
    bool boundary = false; 
    if (file.is_open()) {
        // Read each line from the file and store it in the
        // 'line' variable.
        while (getline(file, line)) {
            std::vector<float> parts = split_string_into_array(line, " ");
            if (count == 0){
                n_entities = parts[0];
                entity_size = parts[1]
                attribute = parts[2];
                boundary = parts[3];
            }
            else {
                int idx = parts[0];
                int attribute_val = -1;
                int boundary_val = -1;
                Point p = {parts[1], parts[2], parts[3]};
                if (parts.size() == 5){
                    if (attribute){
                        attribute_val = parts[4];

                    }
                    if (boundary){
                        boundary_val = parts[4];
                    }
                else if (parts.size() == 6){
                    attribute_val = parts[4];
                    boundary_val = parts[4];
                }
            }
            output_nodes[p] = {idx, attribute_val, boundary_val};
            count ++;
        }

        // Close the file stream once all lines have been
        // read.
        file.close();
    }
    else {
        // Print an error message to the standard error
        // stream if the file cannot be opened.
        std::cerr << "Unable to open file!" << endl;
    }

    return output_nodes;

}
}

void merge_tetgen_nodes(std::vector<<std::string>> nodes_files, std::string nodes_file){

}

void renumber_tetgen_nodes(std::map<int, int> old_to_new){

}

void renumber_tetgen_faces(std::map<int, int> old_to_new);

void renumber_tetgen_tets(std::map<int, int> old_to_new);

std::map<Triangle, std::vector<int> > read_tetgen_faces_to_map(std::string faces_file){
    // remove lines begining with #
    std::map<Triangle, std::vector<int>> output_faces;
    strip_leading_character(nodes_file, "#");
    ifstream file(faces_file);
    std::string line;
    int count = 0;
    int n_entities;
    int entity_size = 3;
    bool attribute = false;
    bool boundary = false; 
    if (file.is_open()) {
        // Read each line from the file and store it in the
        // 'line' variable.
        while (getline(file, line)) {
            std::vector<float> parts = split_string_into_array(line, " ");
            if (count == 0){
                n_entities = parts[0];
                entity_size = parts[1]
                attribute = parts[2];
                boundary = parts[3];
            }
            else {
                int idx = parts[0];
                int attribute_val = -1;
                int boundary_val = -1;
                Triangle f = {parts[1], parts[2], parts[3]};
                if (parts.size() == 5){
                    if (attribute){
                        attribute_val = parts[4];

                    }
                    if (boundary){
                        boundary_val = parts[4];
                    }
                else if (parts.size() == 6){
                    attribute_val = parts[5];
                    boundary_val = parts[6];
                }
            }
            output_faces[f] = {idx, attribute_val, boundary_val};
            count ++;
        }

        // Close the file stream once all lines have been
        // read.
        file.close();
    }
    else {
        // Print an error message to the standard error
        // stream if the file cannot be opened.
        std::cerr << "Unable to open file!" << endl;
    }

    return output_faces;

}
}

void merge_tetgen_faces(std::vector<<std::string>> node_files, std::string faces_files){

}

std::map<Tetrahedron, std::vector<int> > read_tetgen_tets_to_map(std::string tets_file){
    // remove lines begining with #
    std::map<Tetrahedron, std::vector<int>> output_tets;
    strip_leading_character(nodes_file, "#");
    ifstream file(nodes_file);
    std::string line;
    int count = 0;
    int n_entities;
    int entity_size = 4;
    bool attribute = false;
    bool boundary = false; 
    if (file.is_open()) {
        // Read each line from the file and store it in the
        // 'line' variable.
        while (getline(file, line)) {
            std::vector<float> parts = split_string_into_array(line, " ");
            if (count == 0){
                n_entities = parts[0];
                entity_size = parts[1]
                attribute = parts[2];
                boundary = parts[3];
            }
            else {
                int idx = parts[0];
                int attribute_val = -1;
                int boundary_val = -1;
                Tetrahedron t = {parts[1], parts[2], parts[3], parts[4]};
                if (parts.size() == 6){
                    if (attribute){
                        attribute_val = parts[5];

                    }
                    if (boundary){
                        boundary_val = parts[5];
                    }
                else if (parts.size() == 7){
                    attribute_val = parts[5];
                    boundary_val = parts[6];
                }
            }
            output_tets[t] = {idx, attribute_val, boundary_val};
            count ++;
        }

        // Close the file stream once all lines have been
        // read.
        file.close();
    }
    else {
        // Print an error message to the standard error
        // stream if the file cannot be opened.
        std::cerr << "Unable to open file!" << endl;
    }

    return output_tets;

}
}

void merge_tetgen_tets(std::vector<<std::string>> tets_files, std::string tets_files){

}

void write_nodes_to_file(std::map<std::vector<float>, int>& nodes, std::string output_nodes_file){
     std::ofstream outputFile(output_nodes_file);
     outputFile << nodes.size() << " " << 3 << " " << 0 << " " << 0 << "\n";
     if (outputFile.is_open()) {
        for (const auto& pair : *nodes) {
            outputFile << pair.second[0] << " " << pair.first[0] << " " << pair.first[1] << " " << pair.first[2] << "\n";
        }
        outputFile.close();
        std::cout << "Data successfully written to " + std::format("{}", output_nodes_file) << std::endl;
    } else {
        std::cerr << "Error: Unable to open the file for writing." << std::endl;
    }
}

void write_faces_to_file(std::map<Triangle, std::vector<int> >& faces, std::string output_faces_file){
    std::ofstream outputFile(output_faces_file);
     outputFile << *faces.size() << " " << 3 << " " << 0 << " " << 0 << "\n";
     if (outputFile.is_open()) {
        for (const auto& pair : *faces) {
            outputFile << pair.second[0] << " " << pair.first[0] << " " << pair.first[1] << " " << pair.first[2] << "\n";
        }
        outputFile.close();
        std::cout << "Data successfully written to " + std::format("{}", output_faces_file) << std::endl;
    } else {
        std::cerr << "Error: Unable to open the file for writing." << std::endl;
    }
}

void write_tets_to_file(std::map<Tetrahedron, std::vector<int>>& tets, std::string output_tets_file){
    std::ofstream outputFile(output_tets_file);
     outputFile << *tets.size() << " " << 4 << " " << 0 << " " << 0 << "\n";
     if (outputFile.is_open()) {
        for (const auto& pair : *tets) {
            if (pair.second.size() == 0){
                outputFile << pair.second[0] << " " << pair.first[0] << " " << pair.first[1] << " " << pair.first[2] << " " << pair.first[3] << "\n";
            }
            else {
                outputFile << pair.second[0] << " " << pair.first[0] << " " << pair.first[1] << " " << pair.first[2] << " " << pair.first[3] << " " << pair.second[1] << "\n";
            }
        }
        outputFile.close();
        std::cout << "Data successfully written to " + std::format("{}", output_faces_file) << std::endl;
    } else {
        std::cerr << "Error: Unable to open the file for writing." << std::endl;
    }
}
