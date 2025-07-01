#include <stdio.h>
#include <math.h>

__global__ void tiledMatMul(float *a, float *b, float *c, int n, int tile_size) {
    // twi statically-sized pieces of shared memory
    __shared__ int A[]
}

void mat_init(float* mat, int size) {
    for (int i = 0; i < size * size; i++) {
        mat[i] = 1.0;
    }
}


int main() {

    int n = 1 << 10;

    size_t bytes = n * n * sizeof(float);

    float *h_a, *h_b, *h_c;

    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    dim3 THREADS(32, 32);
    dim3 BLOCKS((n + 31) / 32, (n + 31) / 32);


    mat_init(h_a, n);
    mat_init(h_b, n);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    matmul_parallel_square<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 1500; i++) {
        printf("c[%d] = %f\n", i, h_c[i]);
    }

    return 0;
}