#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

vector<int> makeTestSizes(int start, int end) {
    vector<int> sizes;
    for (int n = start; n <= end; n += 256)
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

// tiled shared‑memory kernel
const int TILE = 16;
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

int main() {
    auto sizes = makeTestSizes(1024, 16384);
    std::ofstream csv("benchmark.csv");
    csv << "N,naive_ms,tiled_ms\n";

    for (int N : sizes) {
        size_t bytes = size_t(N) * N * sizeof(int);

        // host data
        vector<int> hA(N * N), hB(N * N), hC(N * N);
        generate(hA.begin(), hA.end(), []() { return rand() % 100; });
        generate(hB.begin(), hB.end(), []() { return rand() % 100; });

        // device allocations
        int *dA, *dB, *dC;
        cudaMalloc(&dA, bytes);
        cudaMalloc(&dB, bytes);
        cudaMalloc(&dC, bytes);
        cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice);

        dim3 threads(TILE, TILE);
        dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

        // timing events
        cudaEvent_t start, stop;
        float ms;

        // naive
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        matMulNaive<<<blocks, threads>>>(dA, dB, dC, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        float t_naive = ms;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // tiled
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        matMulTiled<<<blocks, threads>>>(dA, dB, dC, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        float t_tiled = ms;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // record & report
        csv << N << "," << t_naive << "," << t_tiled << "\n";
        cout << "N = " << N << " | naive = " << t_naive << " ms, tiled = " << t_tiled << " ms\n";

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    csv.close();
    cout << "Done. Results in benchmark.csv\n";
    return 0;
}
