#include "merge_tetgen_outputs.h"


int main(int argc, char** argv){
    return 0;
}

std::map<std::pair<std::vector<float>, int>> read_tetgen_nodes_to_map(std::string nodes_file){

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
