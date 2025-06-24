#include <stdio.h>

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
    cudaMalloc()




    return 0;
}