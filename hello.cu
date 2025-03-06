#include <iostream>

__global__ void helloCUDA() {
    printf("Hello CUDA from GPU! \n");
}

int main() {
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}