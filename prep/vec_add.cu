#include <stdio.h>
#include <math.h>

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

__global__ void vectorInit(float* vec, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        vec[index] = (index * 17 + 13) / 100;
    }
}

int main() {
    int n = 1 << 16; 

    size_t bytes = n * sizeof(float);

    // first initialize pointers
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);


    int numberOfBlocks = 64;
    int numberOfThreadsPerBlock = 256;

    vectorInit<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_a, n);
    vectorInit<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_b, n);
    
    vectorAdd<<<numberOfBlocks, numberOfThreadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 15; i++) {
        printf("c[%d] = %d = %d + %d", i, h_c[i], h_a[i], h_b[i]);
    }

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
