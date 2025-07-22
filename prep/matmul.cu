#include <stdio.h>
#include <math.h>

__global__ void matmul_parallel_square(float* a, float* b, float* c, int n) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    float sum = 0;

    if ((row < n) && (col < n)) {
        // Iterate over row and column:
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void matmul_parallel(float* a, float* b, float* c, int n, int k, int m) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    float sum = 0;

    if ((row < n) && (col < m)) {
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * m + col];
        }
        c[row * m + col] = sum;
    }
}


void matmul_seq(float* a, float* b, float* c, int n, int k,int m) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < m; col++) {
            float sum = 0;
            for (int i = 0; i < k; i++) {
                sum += a[row * k + i] * b[m * i + col];
            }
            c[row * m + col] = sum;
        }
    }
}



void matmul_seq_square(float* a, float* b, float* c, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col ++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += a[row * n + k] * b[k * n + col];
            }
            c[row * n + col] = sum;
        }
    }
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

    for(int i = 0; i < 150; i++) {
        printf("c[%d] = %f\n", i, h_c[i]);
    }

    return 0;
}