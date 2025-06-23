#include <stdio.h>

__global__ void whoami() {

    printf("Thead: (%d, %d, %d) | Block: (%d, %d)\n", 
        threadIdx.x, threadIdx.y, threadIdx.z, 
        blockIdx.x, blockIdx.y);
}

int main() {
    const int b_x = 2, b_y = 2;
    const int t_x = 2, t_y = 2, t_z = 2;

    //int blocks_per_grid = b_x * b_y;
    //int threads_per_block = t_x * t_y * t_z;

    dim3 blocksPerGrid(b_x, b_y);
    dim3 threadsPerBlock(t_x, t_y, t_z);

    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
    

    return 0;
}