#include <algorithm> 
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <filesystem>
#include <hdf5.h>
#include <iostream>
#include <initializer_list>
#include <map>
#include <mpi.h>
#include "nurbs.h"
#include <omp.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <ranges>
#include <vector>

#define TETR_H5_FILE "tetr.h5"
#define TETR_XDMF_FILE "tetr.xdmf"
#define TRIA_H5_FILE "tria.h5"
#define TRIA_XDMF_FILE "tria.xdmf"
#define INVALID -1
#define NUM_THREADS 8

namespace po = boost::program_options;
namespace fs = boost::filesystem;

int main(int argc, char* argv[]){
    std::filesystem::path mesh_parent_folder_path, mesh_folder_path;
    int phase, LX, LY, LZ;
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Creates tetrahedral and triangle mesh in xdmf format from input voxels")
    ("mesh_parent_folder_path,MESH_PARENT_FOLDER_PATH", po::value<std::filesystem::path>(&mesh_parent_folder_path)->required(),  "mesh folder path")
    ("LX", po::value<int>(&LX)->required(),  "length along X in pixel units")
    ("LY", po::value<int>(&LY)->required(),  "length along Y in pixel units")
    ("LZ", po::value<int>(&LZ)->required(),  "length along Z in pixel units, number of image files is LZ + 1")
    ("phase,PHASE", po::value<int>(&phase)->required(),  "phase to reconstruct volume for");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    omp_set_num_threads(NUM_THREADS);
}