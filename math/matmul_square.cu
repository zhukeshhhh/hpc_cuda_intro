#include <stdio.h>
#include <math.h>

__global__ void matrixMulGpu(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // each thread's row
    int col = blockIdx.x * blockDim.x + threadIdx.x; // each thread's col

    int temp_sum = 0;
    if ((row < n) && (col < n)) {
        // iterate over row and down column
        for (int k = 0; k < n; k++) {
            temp_sum += a[row * n + k] * b[k * n + col];
        }
        // assign result
        c[row * n + col] = temp_sum;
    }
}

void matrixMulCpu(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int temp_sum = 0;
            for (int k = 0; k < n; k++) {
                temp_sum += a[i * n + k] * b[k * n + j];
            }
            c[j + i * n] = temp_sum;
        }
    }
}

void init_matrix(int *m, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[i * n + j] = rand() % 100;
        }
    }
}

int main() {

    int n  = 1 << 10; // matrix size of 1024 (2^10)

    size_t bytes = n * n * sizeof(int);

    int *h_a, *h_b, *h_c; // host pointers

    // allocate host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    //device pointers
    int *d_a, *d_b, *d_c;

    //allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // init matrices
    init_matrix(h_a, n);
    init_matrix(h_b, n);


    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);


    int BLOCK_SIZE = 16; // threads per one block
    int GRID_SIZE = (int)ceil(n / BLOCK_SIZE); // block in each dimension

    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    matrixMulGpu<<<grid, threads>>>(d_a, d_b, d_c, n); // launch kernel

    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    int *h_c_cpu = (int*)malloc(bytes);
    matrixMulCpu(h_a, h_b, h_c_cpu, n);

    // After copying h_c from GPU
    for (int i = 0; i < n * n; i++) {
        if (h_c[i] != h_c_cpu[i]) {
            printf("Mismatch at index %d: GPU %d != CPU %d\n", i, h_c[i], h_c_cpu[i]);
            break;
        }
    }


    for (int i = 0; i < 36; i++) {
        printf("c[%d] = %d\n", i, h_c[i]);   
    }



    printf("SUCCESS!\n");

    return 0;
}