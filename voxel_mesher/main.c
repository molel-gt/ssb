#include <algorithm> 
#include <iostream>
#include <map>
#include <vector>
// #include "entities.h"

// float *a;
// int n;
// err = cudaMalloc(&a, sizeof(float)*n);
// kernel<<<n/32,32>>>(a,...);
// incr(a,n);

// void incr(float* x, int n){
// #pragma acc parallel loop deviceptr(x)
// for (int i = 0; i < n; ++i)
//     x[i] += 1.0f;
// }

struct Voxel {
    int p000, p100, p110, p010, p001, p101, p111, p011;
} Voxel;


int main(int argc, char *argv[]){
    int Nx = 2, Ny = 2, Nz = 2;
    std::map<std::vector<int>, int> voxels;
    std::map<std::vector<int>, int> points;
    std::map<std::vector<int>, int> edges;
    // std::map<std::vector<int>, int> faces;
    // std::map<std::vector<int>, int> tetrahedrons;

    voxels[{0, 0, 0}] = 1;
    voxels[{1, 0, 0}] = 1;
    voxels[{0, 1, 0}] = 1;
    voxels[{1, 1, 0}] = 1;
    voxels[{0, 0, 1}] = 1;
    voxels[{1, 0, 1}] = 1;
    voxels[{0, 1, 1}] = 1;
    voxels[{1, 1, 1}] = 1;
    std::cout << voxels.count({13, 4, 5}) << "\n";

    // build points dictionary
    int num_points = 0;
    for (int i = 0; i < Nx; i++){
        for (int j = 0; j < Ny; j++){
            for (int k = 0; k < Nz; k++){
                if (voxels[{i, j, k}] == 1){
                    points[{i, j, k}] = num_points;
                    num_points ++;
                }
            }
        }
    }

    // build edges
    // id of points corresponding to unit cube
    int p000, p100, p110, p010, p001, p101, p111, p011;
    int num_edges = 0;
    for (int i = 0; i < Nx; i++){
        for (int j = 0; j < Ny; j++){
            for (int k = 0; k < Nz; k++){
                int value = points[{i, j, k}];
                if (points.count({i, j, k}) > 0){
                    std::vector<std::vector<int>> neighbors{{i + 1, j, k}, {i, j + 1}, {i, j, k + 1}, {i - 1, j, k}, {i, j - 1, k}, {i, j, k - 1}};
                    for (int idx = 0; idx < 6; idx ++){
                        int value1 = points[neighbors[idx]];
                        std::cout << points.count(neighbors[idx]) << "\n";
                        if (points.count(neighbors[idx]) > 0)
                        {
                            std::vector edge = {value, value1};
                            std::sort(edge.begin(), edge.end()); 
                            edges[edge] = num_edges;
                            num_edges ++;
                        }
                    }
                }
            }
        }
    }
    // build tetrahedrons and faces
    std::vector<std::vector<int>> cube;
    std::map<int, int> cube_points;
    std::vector<std::vector<int>> faces;
    std::vector<std::vector<int>> tetrahedrons;
    std::vector<int> tet;
    int invalid = -1;

    for (int i = 0; i < Nx; i++){
        for (int j = 0; j < Ny; j++){
            for (int k = 0; k < Nz; k++){
                cube = {
                    {i, j, k},
                    {i + 1, j, k},
                    {i + 1, j + 1, k},
                    {i, j + 1, k},
                    {i, j, k + 1},
                    {i + 1, j, k + 1},
                    {i + 1, j + 1, k + 1},
                    {i, j + 1, k + 1}
                    };
                for (int idx = 0; idx < 8; idx++){
                    int value = points[{i, j, k}];
                    if (points.count({i, j, k}) > 0){
                        cube_points[idx] = value;
                    } else {
                        cube_points[idx] = invalid;
                    }
                }
                for (int idx = 0; idx < 5; idx ++){
                    switch(idx){
                        case 0:
                        {
                            tet = {cube_points[0], cube_points[1], cube_points[3], cube_points[4]};
                        }
                        case 1:
                        {
                            tet = {cube_points[1], cube_points[1], cube_points[3], cube_points[6]};
                        }
                        case 2:
                        {
                            tet = {cube_points[4], cube_points[5], cube_points[6], cube_points[1]};
                        }
                        case 3:
                        {
                            tet = {cube_points[4], cube_points[7], cube_points[6], cube_points[3]};
                        }
                        case 4:
                        {
                            tet = {cube_points[0], cube_points[1], cube_points[3], cube_points[4]};
                        }
                        default:
                        {
                            tet = {invalid, invalid, invalid, invalid};
                        }
                        std::sort(tet.begin(), tet.end());
                        if (std::find(tet.begin(), tet.end(), invalid) != tet.end()){
                            tetrahedrons.push_back(tet);
                        }
                    }
                }
            }
        }
    }

    return 0;
}