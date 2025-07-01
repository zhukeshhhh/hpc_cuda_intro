#include <stdio.h>

__global__ void kernel1(int* a, int N) {
    int i = threadIdx.x * blockDim.x + blockIdx.x;

    if (i < N) {
        a[i] = blockDim.x;
    }
    
}

__global__ void kernel2(int* a, int N) {
    int i = threadIdx.x * blockDim.x + blockIdx.x;
    if (i < N) {
        a[i] = threadIdx.x;
    }
    
}

__global__ void kernel3(int* a, int N) {
    int i = threadIdx.x * blockDim.x + blockIdx.x;
    if (i < N) {
        a[i] = i;
    }
}

int main() {
    int N = 10; // number of elements in array
    size_t size = N * sizeof(int);

    // allocate host memory
    int* h_a = (int*)malloc(size);

    // allocate device memory
    int * d_a;
    cudaMalloc((void**)&d_a, size);

    // configuration
    int threadsPerBlock = 32;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    //kernel1<<<3, 4>>>(d_a, N);
    //kernel2<<<numBlocks, threadsPerBlock>>>(d_a, N);
    kernel3<<<numBlocks, threadsPerBlock>>>(d_a, N);

    // copy result to the host
    cudaMemcpy( h_a, d_a, size, cudaMemcpyDeviceToHost);



    for (int i = 0; i < N; i++) {
        printf("h_a[%d] = %d\n", i, h_a[i]);
    }


    cudaFree(d_a);
    free(h_a);
    
    return 0;
}
