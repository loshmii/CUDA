#include <stdio.h>
#include <iostream>
#include <chrono>

__global__ void vector_add(int* d_a, int* d_b, int* d_c, int N) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx < N) d_c[idx] = d_a[idx] + d_b[idx];
}

void vector_add_cpu(int* a, int*b, int*c, int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int N = 1 << 20;
    int* a, * b, * c;
    a = (int*)malloc(N*sizeof(int));
    b = (int*)malloc(N*sizeof(int));
    c = (int*)malloc(N*sizeof(int));
    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, N*sizeof(int));
    cudaMalloc(&d_b, N*sizeof(int));
    cudaMalloc(&d_c, N*sizeof(int));
    for (int i = 0; i < N; i++) {
        a[i] = i/2;
        b[i] = i/2;
    }
    cudaMemcpy(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
    const int threads_per_block = 256;
    const int num_of_blocks = (N + threads_per_block - 1) / threads_per_block;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    vector_add<<<num_of_blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    vector_add_cpu(a,b,c,N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = gpu_end - gpu_start;
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
    if (cpu_time < gpu_time) throw 1;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}