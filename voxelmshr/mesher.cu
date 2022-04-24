#include <iostream>
#include <set>

#include <gmsh.h>
#include <algorithm>

using namespace std;
using namespace gmsh;

// default grid size
const int Nx = 101;
const int Ny = 201;
const int Nz = 101;
const int NUM_GRID = Nx * Ny * Nz;

struct Rectangle {
    int x0 = 0;
    int y0 = 0;
    int z0 = 0;
    int dx = 0;
    int dy = 0;
    int dz = 0;
};

__global__ void makeRectangles(int *voxelData, struct Rectangle *rectangles, int NX)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < NX){
        for (int i_y = 0; i_y < Ny - 2; i_y++){
            int squares[Nz];
            for (int i_z = 0; i_z < Nz; i_z++){
                int idx_p0 = idx + i_y * Ny + i_z * Ny * Nz;
                int idx_p1 = idx + 1 + i_y * Ny + i_z * Ny * Nz;
                int idx_p2 = idx + (i_y + 1) * Ny + i_z * Ny * Nz;
                int idx_p3 = idx + 1 + (i_y + 1) * Ny + i_z * Ny * Nz;
                if (voxelData[idx_p0] && voxelData[idx_p1] && voxelData[idx_p2] && voxelData[idx_p3]){
                    squares[i_z] = 1;
                }
                else {
                    squares[i_z] = 0;
                }
            }

            int partLength = 0;
            int startPos = 0;
            int rect_counter = 0;

            for (int i_z = 0; i_z < Nz - 1; i_z++){
                if (squares[i_z] && squares[i_z + 1]){
                    partLength++;
                }
                else if (squares[i_z] && !squares[i_z + 1]) {
                    int rect_idx = idx + i_y * Ny + i_z * Ny * Nz;
                    rectangles[rect_idx].x0 = idx;
                    rectangles[rect_idx].y0 = i_y;
                    rectangles[rect_idx].z0 = startPos;
                    rectangles[rect_idx].dx = 1;
                    rectangles[rect_idx].dy = 1;
                    rectangles[rect_idx].dz = partLength;
                    rect_counter ++;
                    startPos = i_z + 1;
                    partLength = 0;
                }
                else {
                    partLength = 0;
                    startPos = i_z + 1;
                    
                }
                if (i_z == Nz - 2 && partLength){
                    int rect_idx = idx + i_y * Ny + i_z * Ny * Nz;
                    rectangles[rect_idx].x0 = idx;
                    rectangles[rect_idx].y0 = i_y;
                    rectangles[rect_idx].z0 = startPos;
                    rectangles[rect_idx].dx = 1;
                    rectangles[rect_idx].dy = 1;
                    rectangles[rect_idx].dz = partLength;
                    rect_counter ++;
                    startPos = i_z + 1;
                }
            }

        } 
    }
}

void generateMesh(struct Rectangle *rectangles){
    gmsh::initialize();
    gmsh::logger::start();
    
    gmsh::model::add("porous");
    std::vector<std::pair<int, int> > solids;
    std::vector<std::pair<int, int> > ov;
    std::vector<std::vector<std::pair<int, int> > > ovv;
    int tag = 0;
    try
    {
        tag++;
        gmsh::model::occ::addBox(0, 0, 0, Nx - 1, Ny - 1, Nz - 1, tag);
    }
    catch (...){
        gmsh::logger::write("Could not create OpenCASCADE shapes!");
        return;
    }

    for (int idx = 0; idx < Nx - 1; idx++){
        for (int idy = 0; idy < Ny - 1; idy++){
            for (int idz = 0; idz < Nz - 1; idz++){
                int rect_index = idx + idy * Ny + idz * Ny * Nz;
                if (3 > rectangles[rect_index].dx > 0 && 3 > rectangles[rect_index].dy > 0 && 3 > rectangles[rect_index].dz > 0){
                    tag++;
                    printf("%d,%d,%d\n", rectangles[rect_index].dx, rectangles[rect_index].dy, rectangles[rect_index].dz);
                    gmsh::model::occ::addBox(rectangles[rect_index].x0, rectangles[rect_index].y0, rectangles[rect_index].z0,
                                            rectangles[rect_index].dx, rectangles[rect_index].dy, rectangles[rect_index].dz,
                                            tag
                                            );
                    solids.push_back({3, tag});
                }
            }
        }
    }

    gmsh::model::occ::cut({{3, 1}}, solids, ov, ovv, tag + 1);
    gmsh::model::occ::synchronize();

    gmsh::model::addPhysicalGroup(3, {tag + 1}, 1);

    double lcar1 = .1;
    gmsh::model::getEntities(ov, 0);
    gmsh::model::mesh::setSize(ov, lcar1);

    gmsh::model::mesh::generate(3);
    gmsh::logger::write("Writing mesh..");
    gmsh::write("../porous.msh");
    
    gmsh::finalize();
}

#define N (2048*2048)
#define THREADS_PER_BLOCK 512
int main(int argc, char **argv)
{
    int *voxels = (int *)malloc(sizeof(int) * NUM_GRID);
    struct Rectangle *rectangles = (struct Rectangle *)malloc(sizeof(struct Rectangle) * NUM_GRID);
    int *d_voxels;
    struct Rectangle *d_rectangles;

    cudaMalloc((void **)&d_voxels, sizeof(int) * NUM_GRID);
    cudaMalloc((void **)&d_rectangles, sizeof(struct Rectangle) * NUM_GRID);

    voxels = (int *)malloc(sizeof(int) * NUM_GRID);
    rectangles = (struct Rectangle *)malloc(sizeof(struct Rectangle) * NUM_GRID);
    // build voxels
     for (int i = 0; i < 30; i++){
        voxels[i] = 1;
    }

    // copy inputs to device
    cudaMemcpy(d_voxels, voxels, sizeof(int) * NUM_GRID, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rectangles, rectangles, sizeof(struct Rectangle) * NUM_GRID, cudaMemcpyHostToDevice);
    makeRectangles<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_voxels, d_rectangles, Nx);
    cudaMemcpy(rectangles, d_rectangles, sizeof(struct Rectangle) * NUM_GRID, cudaMemcpyDeviceToHost);

    // free memory on device
    cudaFree(d_voxels); cudaFree(d_rectangles);
    // build geometry
    // generateMesh(rectangles);
    printf("%d\n", rectangles[0].dz);
    return 0;
}