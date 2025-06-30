#include <stdio.h>
#include <math.h>

__global__ void matmul_parallel() {
    
}


void matmul_seq(float* a, float* b, float* c, int n, int k,int m) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < m; col++) {
            float sum = 0;
            for (int i = 0; i < k; i++) {
                sum += a[row * k + i] * b[k * i + col];
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

    int size = 100;

    size_t bytes = size * size * sizeof(float);

    float *a, *b, *c;

    a = (float*)malloc(bytes);
    b = (float*)malloc(bytes);
    c = (float*)malloc(bytes);


    mat_init(a, size);
    mat_init(b, size);


    matmul_seq(a, b, c, size, size, size);


    for (int i = 0; i < 100; i++) {
        printf("c[%d] = %f | a = %f | b = %f \n", i, c[i], a[i], b[i]);
    }


    return 0;
}