#include <stdio.h>
#include <math.h>

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

__global__ void vectorInit(float* vec, int n) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < n) {
        vec[index] = (index * 17 + 13) % 100;
    }
}

int main() {

    int id = cudaGetDevice(&id);

    int n = 1 << 16;

    size_t bytes = n * sizeof(float);

    float *a, *b, *c;

    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    int numberOfBlocks = 64;
    int numberOfThreadsPerBlock = 1024;

    vectorInit<<<numberOfBlocks, numberOfThreadsPerBlock>>>(a, n);
    vectorInit<<<numberOfBlocks, numberOfThreadsPerBlock>>>(b, n);
    
    vectorAdd<<<numberOfBlocks, numberOfThreadsPerBlock>>>(a, b, c, n);

    cudaDeviceSynchronize();

    for (int i = 0; i < n; i++) {
        printf("c[%d] = %f = %f + %f\n", i, c[i], a[i], b[i]);
    }
}
