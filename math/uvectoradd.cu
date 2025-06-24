#include <stdio.h>
#include <math.h>

void init_vector(int* vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (rand() * rand()) % 100;
    }
}

__global__ void vector_addUM(int *a, int *b, int *c, int n) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x; // calculate global thread id
    if (tid < n) { // thread out of bound of allocated memory
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int id = cudaGetDevice(&id); // get device id for other CUDA calls
    int n = 1 << 16; // 65536 elements in array
    size_t bytes = n * sizeof(int);

    // declare unified memory pointers
    int *a, *b, *c;

    // allocate memory for these pointers
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // init vectors
    init_vector(a, n);
    init_vector(b, n);

    // block size:
    int NUM_THREADS = 256;

    // grid size:
    int NUM_BLOCKS = (int)ceil(n / NUM_THREADS);


    // prefetching data from memory to gpu
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);

    vector_addUM<<<NUM_BLOCKS, NUM_THREADS>>>(a, b, c ,n); // launch kernel

    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    printf("First 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = a[%d] + b[%d]\n", c[i], a[i], b[i]);
    }

    printf("COMPLETED SUCCESFULLY\n");

    return 0;
}