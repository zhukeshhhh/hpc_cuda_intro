#include <stdio.h>
#include <math.h>

// DRAM is slow. Need fewer memory stalls.
// Solution: Shared Memory -> User-Managed L1 cache. Private for each threadblock
// 5-10 cycles of access time instead of hundreds in DRAM

// Put pieces of large input in cache

#define SHMEM_SIZE 16 * 16 * 4

__global__ void tiledMatrixMul(int *a, int *b, int *c, int n, int tile_size) {
    // two statically sized pieces of shared memory
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int by = blockIdx.x;
    int by = blockIdx.y;

    // calculate global row and column positions for this thread
    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;

    // intermediate sum for element being written
    int temp_val = 0;

    // sweep tiles over rntire matrix
    for (int i = 0; i < (n / tile_size); i++) {
        
    }
       
}


int main() {


    return 0;
}