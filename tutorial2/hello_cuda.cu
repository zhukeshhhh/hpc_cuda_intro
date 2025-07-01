#include <stdio.h>

__global__ void hello_cuda() {
    printf("Hello Cuda\n");
    printf("Block Index X: %d, Block Index Y: %d, Thread Index X: %d, Thread INdex Y: %d\n",
        blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

int main(int argc, char **argv) {
    hello_cuda<<<2, 2>>>();
    cudaDeviceSynchronize();
    return 0;
}