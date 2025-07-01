#include <stdio.h>
#include <math.h>

void init_vector(int* vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = rand() % 100;
    }
}

__global__ void vector_add(int *a, int *b, int *c, int n) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; // calculate global thread id
    if (tid < n) { // thread out of bound of allocated memory
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int n = 1 << 16; // 2^16 (65536 elements)
    int *h_a, *h_b, *h_c; // host pointers
    int *d_a, *d_b, *d_c; // device pointers

    size_t bytes = sizeof(int) * n;

    // allocate host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // initialize vectors with values from 0 to 99
    init_vector(h_a, n);
    init_vector(h_b, n);

    // copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // block size:
    int NUM_THREADS = 256;

    // grid size:
    int NUM_BLOCKS = (int)ceil(n / NUM_THREADS);

    // launch kernel on default stream w/o shared memory
    vector_add<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize(); // make sure it's finished before cudaMemcpy

    //copy sum vector from device to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("First 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = a[%d] + b[%d]\n", h_c[i], h_a[i], h_b[i]);
    }


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    printf("COMPLETED SUCCESFULLY\n");

    return 0;
}