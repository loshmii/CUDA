#include <stdio.h>
#include <iostream>

__global__ void parallel_sum(int* arr, long long* sum, int N) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int val = (idx < N) ? arr[idx] : 0;
    extern __shared__ unsigned long long partial[];
    for (unsigned int delta = 16; delta > 0; delta >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, delta);
    }

    if ((threadIdx.x & 31) == 0) {
        partial[threadIdx.x/32] = val;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long res = 0;
        for (int i = 0; i < (blockDim.x + 31)/32; i++) {
            res += partial[i];
        }
        atomicAdd((unsigned long long*) sum, res);
    }
}

int main() {
    int N = 1e6;
    int *arr;
    long long *sum;
    sum = (long long*) malloc(sizeof(long long));
    *sum = 0;
    size_t bytes = N*sizeof(int);
    arr = (int*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        arr[i] = i % 13;
    }

    int* d_arr;
    long long* d_sum;
    cudaMalloc(&d_arr, bytes);
    cudaMalloc(&d_sum, sizeof(long long));
    cudaMemset(d_sum, 0, sizeof(long long));
    cudaMemcpy(d_arr, arr, bytes, cudaMemcpyHostToDevice);
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1)/blockDim.x);
    size_t sharedMemSize = ((blockDim.x + 31)/32) * sizeof(unsigned long long);

    parallel_sum<<<gridDim, blockDim, sharedMemSize>>>(d_arr, d_sum, N);
    cudaDeviceSynchronize();
    cudaMemcpy(sum, d_sum, sizeof(long long), cudaMemcpyDeviceToHost);
    std::cout << *sum << std::endl;

    cudaFree(d_arr);
    cudaFree(d_sum);
    free(arr);
    free(sum);


    return 0;
}