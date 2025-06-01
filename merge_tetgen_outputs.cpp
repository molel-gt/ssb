#include "merge_tetgen_outputs.h"


int main(int argc, char** argv){
    return 0;
}

void strip_leading_character(std::string input_file, char character){
    std::system("sed -i '/#/d'" + " " + std::format("{}", input_file));
}

template <typename std::pair<T, V> split_string_into_pair(char[] text, char delimiter, int size_key){
    char *token = strtok(text, delimiter);
    std::pair<T, V> output;
    int count = 0;
    T key;
    V value;
    while (token != NULL)
    {
        if ( count < size_key){
            key[count] = std::stof(token);
        }
        else {
            value[count - size_key] = std::stof(token);
        }
        count ++;
        token = strtok(NULL, delimiter);
    }
    output.first = key;
    output.second = value;

    return output;
    
}

std::map<std::pair<std::vector<float>, int>> read_tetgen_nodes_to_map(std::string nodes_file){
    // remove lines begining with #
    strip_leading_character(nodes_file, "#");
    ifstream file(nodes_file);
    std::string line;
    if (file.is_open()) {
        // Read each line from the file and store it in the
        // 'line' variable.
        while (getline(file, line)) {
            if (line.at(0) == "#"){
                continue
            }
            else {

            }
            std::cout << line << std::endl;
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

}

void merge_tetgen_nodes(std::vector<<std::string>> nodes_files, std::string nodes_file){

}

void renumber_tetgen_faces(std::map<int, int> old_to_new){

}

std::map<std::vector<int>, int> read_tetgen_faces_to_map(std::string faces_file){

}

void merge_tetgen_faces(std::vector<<std::string>> node_files, std::string faces_files){

}

std::map<std::vector<int>, int> read_tetgen_tets_to_map(std::string tets_file){

}

void merge_tetgen_tets(std::vector<<std::string>> tets_files, std::string tets_files){

}

void write_nodes_to_file(std::map<std::vector<float>, int>& nodes, std::string output_nodes_file){

}

void write_faces_to_file(std::map<std::vector<float>, int>& faces, std::string output_faces_file){

}

void write_tets_to_file(std::map<std::vector<float>, int>& tets, std::string output_tets_file){

}
