#include <stdio.h>
#include <math.h>
#include <iostream>

unsigned int nextpowof2(unsigned int val) {
    val--;
    val |= val >> 1;
    val |= val >> 2;
    val |= val >> 4;
    val |= val >> 8;
    val |= val >> 16;
    return val + 1;
}

__global__ void max_reduction_one_block(float* arr, float* dest, int col_dim) {
    float val = (threadIdx.x < col_dim) ? arr[blockIdx.x * col_dim + threadIdx.x] : -INFINITY;
    extern __shared__ float row_max[];
    for (unsigned int delta = 16; delta > 0; delta /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, delta));
    }
    __syncthreads();

    if ((threadIdx.x & 31) == 0) {
        row_max[threadIdx.x/32] = val;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < (blockDim.x + 31)/32; i++) {
            row_max[0] = max(row_max[0], row_max[i]);
        }
    }
    __syncthreads();

    if (threadIdx.x < col_dim) dest[blockIdx.x * col_dim + threadIdx.x] = arr[blockIdx.x * col_dim + threadIdx.x] - row_max[0];
}

__global__ void softmax_one_block(float* logits, int col_dim) {
    float val = (threadIdx.x < col_dim) ? exp(logits[blockIdx.x * col_dim + threadIdx.x]) : 0;
    float exp_val = val;
    extern __shared__ float partial[];
    for (unsigned int delta = 16; delta > 0; delta /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, delta);
    }
    if (threadIdx.x == 0) {
        for (int i = 0; i < (blockDim.x + 31)/32; i++) {
            partial[i] = 0;
        }
    }
    __syncthreads();


    if ((threadIdx.x & 31) == 0) {
        partial[threadIdx.x/32] = val;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 1; i < (blockDim.x + 31)/32; i++) {
            partial[0] += partial[i];
        }
    }
    __syncthreads();

    if (threadIdx.x < col_dim) logits[blockIdx.x * col_dim + threadIdx.x] = exp_val / partial[0];
}

int main() {
    int rows = 16;
    int cols = 256;
    float* logits;
    size_t bytes = rows*cols * sizeof(float);
    logits = (float*) malloc(bytes);
    for (int i = 0; i < rows*cols ; i++) {
        logits[i] = (float) (i % 1001 * 1.0)/100000;
    }
    float* d_logits, * softmax_device, * softmax_host;
    cudaMalloc(&d_logits, bytes);
    cudaMalloc(&softmax_device, bytes);
    cudaMemcpy(d_logits, logits, bytes, cudaMemcpyHostToDevice);
    softmax_host = (float*) malloc(bytes);
    dim3 blockDim(min(1024, nextpowof2(cols)));
    unsigned int blocks_per_row = (cols + 1023) / 1024;
    dim3 gridDim(rows, blocks_per_row);
    if (blocks_per_row == 1) {
        int size = (blockDim.x + 31) / 32;
        max_reduction_one_block<<<gridDim, blockDim, size * sizeof(float)>>>(d_logits, softmax_device, cols);
        cudaDeviceSynchronize();
        softmax_one_block<<<gridDim, blockDim, size * sizeof(float)>>>(softmax_device, cols);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(softmax_host, softmax_device, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << softmax_host[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}