__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    float *h_x, *h_y;
    int n;
    // omitted: allocate CPU memory for h_x and h_y and initialize contents
    float *d_x, *d_y;
    int nblocks = (n + 255) / 256;
    cudaMalloc( &d_x, n * sizeof(float));
    cudaMalloc( &d_y, n * sizeof(float));
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
    saxpy<<<nblocks, 256>>>(n, 2.0, d_x, d_y);
    cudaMemcpy(h_x. d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    // omittted: use h_y on CPU, free memory pointed to by h_x, h_y, d_x, and d_y
}