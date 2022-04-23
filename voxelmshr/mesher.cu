#include <iostream>
#include "gmsh.h"
#include <set>

using namespace std;

// default grid size
const int Nx = 203;
const int Ny = 451;
const int Nz = 801;
const int NUM_GRID = Nx * Ny * Nz;

struct Rectangle {
    int x0;
    int y0;
    int z0;
    int dx;
    int dy;
    int dz;
};

__global__ void makeRectangles(int *voxelData, Rectangle *rectangles, int NX)
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
    voxels[0] = 1;
    voxels[1] = 1;
    voxels[2] = 1;
    voxels[3] = 1;
    voxels[4] = 1;
    voxels[5] = 1;
    voxels[6] = 1;
    voxels[7] = 1;
    voxels[8] = 1;
    voxels[9] = 1;
    voxels[10] = 1;
    voxels[11] = 1;
    voxels[12] = 1;
    voxels[13] = 1;
    voxels[14] = 1;
    voxels[15] = 1;
    voxels[16] = 1;
    voxels[17] = 1;
    voxels[18] = 1;
    voxels[19] = 1;
    voxels[20] = 1;
    voxels[21] = 1;
    voxels[22] = 1;
    voxels[23] = 0;
    voxels[24] = 1;
    voxels[25] = 0;
    voxels[26] = 0;

    // copy inputs to device
    cudaMemcpy(d_voxels, voxels, sizeof(int) * NUM_GRID, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rectangles, rectangles, sizeof(struct Rectangle) * NUM_GRID, cudaMemcpyHostToDevice);
    makeRectangles<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_voxels, d_rectangles, Nx);
    cudaMemcpy(rectangles, d_rectangles, sizeof(struct Rectangle) * NUM_GRID, cudaMemcpyDeviceToHost);

    // free memory on device
    cudaFree(d_voxels); cudaFree(d_rectangles);

    gmsh::initialize(argc, argv);
    gmsh::logger::start();
    
    gmsh::model::add("porous");
    gmsh::model::occ::addBox(0, 0, 0, Nx - 1, Ny - 1, Nz - 1);
    gmsh::model::occ::addSphere(Nx/2, Ny/2, Nz/2, Nx/4);
    std::vector<std::pair<int, int> > ov;
    std::vector<std::vector<std::pair<int, int> > > ovv;

    gmsh::model::occ::cut({{3, 1}}, {{3, 2}}, ov, ovv, 3);
    std::vector<std::pair<int, int> > holes;
    gmsh::model::occ::synchronize();
    double lcar2 = .001;
    gmsh::model::mesh::setSize(ov, lcar2);
    gmsh::model::mesh::generate(3);
    gmsh::write("porous.msh");

    //
    gmsh::finalize();
    return 0;
}