#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <ranges>
#include <set>
#include <stdio.h>
#include <string>
#include <sstream>
#include <typeinfo>
#include <vector>

#include <boost/program_options.hpp>
// #include <omp.h>

typedef std::array<int, 4> Tetrahedron;
typedef std::array<int, 3> Coordinate;
const int N = 778 * 402 * 402;
const int N_tets = 778 * 402 * 402 * 5;

namespace po = boost::program_options;

void split_string_into_array(char* text, const char* delimiter, std::vector<int>& output);

void read_phase_data(std::filesystem::path, std::map<Coordinate, int>&);

std::array<Coordinate, 8> make_cube(const std::map<Coordinate, int>&, Coordinate&, int);

std::array<Tetrahedron, 5> make_tetrahedrons_from_cube(const std::array<int, 8>&);

void write_tetgen_node_file(std::filesystem::path, const std::map<Coordinate, int>&, int, std::array<float, 3>&);

void write_tetgen_ele_file(std::filesystem::path, const std::vector<Tetrahedron>&, int);

bool cube_is_filled(const std::map<Coordinate, int>&, Coordinate&, int);

std::array<int, 8> cube_coords_to_cube_ids(const std::map<Coordinate, int>&, std::array<Coordinate, 8>&);
