#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string>
#include <vector>

typedef std::array<float, 3> Point;
typedef std::array<int, 3> Triangle;
typedef std::array<int, 4> Tetrahedron;

void strip_leading_character(std::filesystem::path input_file, char character);

template <typename std::pair<T, V> split_string_into_pair(char[] text, char delimiter, int size_key);

std::map< Point, std::vector<int> > read_tetgen_nodes_to_map(std::filesystem::path nodes_file);

void merge_tetgen_nodes(std::vector<<std::string>> nodes_files, std::filesystem::path nodes_file);

void renumber_tetgen_nodes(std::map<int, int> old_to_new);

void renumber_tetgen_faces(std::map<int, int> old_to_new);

void renumber_tetgen_tets(std::map<int, int> old_to_new);

std::map<Triangle, std::vector<int> > read_tetgen_faces_to_map(std::filesystem::path faces_file);

void merge_tetgen_faces(std::vector<<std::string>> node_files, std::string faces_files);

std::map<Tetrahedron, std::vector<int> > read_tetgen_tets_to_map(std::filesystem::path tets_file);

void merge_tetgen_tets(std::vector<<std::string>> tets_files, std::string tets_files);

void write_nodes_to_file(const std::map<Point, std::vector<int> >& nodes, std::filesystem::path output_nodes_file);

void write_faces_to_file(const std::map<Triangle, std::vector<int> >& faces, std::filesystem::path output_faces_file);

void write_tets_to_file(const std::map<Tetrahedron, std::vector<int>>& tets, std::filesystem::path output_tets_file);
