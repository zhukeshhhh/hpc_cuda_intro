#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

using std::cout;
using std::generate;
using std::vector;

vector<int> makeTestSizes(int start, int end, int step) {
    vector<int> sizes;
    for (int n = start; n <= end; n += step)
        sizes.push_back(n);
    return sizes;
}

// naive kernel
__global__ void matMulNaive(const int* A, const int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int tmp = 0;
        for (int k = 0; k < N; ++k)
            tmp += A[row * N + k] * B[k * N + col];
        C[row * N + col] = tmp;
    }
}

// tiled kernel
const int TILE = 32;
__global__ void matMulTiled(const int* A, const int* B, int* C, int N) {
    __shared__ int sA[TILE][TILE];
    __shared__ int sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    int tmp = 0;

    int numTiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int aIdx = row * N + t * TILE + threadIdx.x;
        int bIdx = (t * TILE + threadIdx.y) * N + col;
        sA[threadIdx.y][threadIdx.x] = (row < N && (t * TILE + threadIdx.x) < N)
                                        ? A[aIdx] : 0;
        sB[threadIdx.y][threadIdx.x] = (col < N && (t * TILE + threadIdx.y) < N)
                                        ? B[bIdx] : 0;
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            tmp += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = tmp;
}

float benchmarkKernel(void (*kernel)(const int*, const int*, int*, int),
                      const int* dA, const int* dB, int* dC,
                      int N, dim3 blocks, dim3 threads, int runs) {
    float total = 0.0f;
    cudaEvent_t start, stop;

    for (int i = 0; i < runs; i++) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // launch kernel
        kernel<<<blocks, threads>>>(dA, dB, dC, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        if (i > 0) total += ms; // discard first run
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return total / (runs - 1);
}

int main() {
    auto sizes = makeTestSizes(1024, 16384, 256);
    std::ofstream csv("benchmark.csv");
    csv << "N,naive_mean,tiled_mean\n";

    const int numRuns = 20;  // more runs for stability

    for (int N : sizes) {
        size_t bytes = size_t(N) * N * sizeof(int);

        // Allocate pinned memory for consistency
        int *hA, *hB, *hC;
        cudaMallocHost(&hA, bytes);
        cudaMallocHost(&hB, bytes);
        cudaMallocHost(&hC, bytes);

        generate(hA, hA + N * N, []() { return rand() % 100; });
        generate(hB, hB + N * N, []() { return rand() % 100; });

        // Device allocations
        int *dA, *dB, *dC;
        cudaMalloc(&dA, bytes);
        cudaMalloc(&dB, bytes);
        cudaMalloc(&dC, bytes);
        cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

        dim3 threads(TILE, TILE);
        dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

        // Warm-up for this size
        matMulNaive<<<blocks, threads>>>(dA, dB, dC, N);
        matMulTiled<<<blocks, threads>>>(dA, dB, dC, N);
        cudaDeviceSynchronize();

        // Benchmark
        float naive_mean = benchmarkKernel(matMulNaive, dA, dB, dC, N, blocks, threads, numRuns);
        float tiled_mean = benchmarkKernel(matMulTiled, dA, dB, dC, N, blocks, threads, numRuns);

        csv << N << "," << naive_mean << "," << tiled_mean << "\n";
        cout << "N = " << N << " | naive(avg) = " << naive_mean
             << " ms, tiled(avg) = " << tiled_mean << " ms\n";

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        cudaFreeHost(hA);
        cudaFreeHost(hB);
        cudaFreeHost(hC);
    }

    csv.close();
    cout << "Done. Results in benchmark.csv\n";
    return 0;
}
