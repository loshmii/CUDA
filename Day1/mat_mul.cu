#include <iostream>
#include <chrono>
const int tile_size = 16;

__global__ void mat_mul(int* d_a, int* d_b, int* d_c, int a_rows, int a_cols, int b_cols) {
    int numOfTiles = (a_cols + tile_size - 1)/tile_size;
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    __shared__ int sharedA[tile_size][tile_size];
    __shared__ int sharedB[tile_size][tile_size];
    for (int t = 0; t < numOfTiles; t++) {
        sharedA[threadIdx.y][threadIdx.x] = (row < a_rows && t*tile_size + threadIdx.x < a_cols) ? d_a[row * a_cols + t * tile_size + threadIdx.x] : 0; 
        sharedB[threadIdx.y][threadIdx.x] = (t*tile_size + threadIdx.y < a_cols && col < b_cols) ? d_b[(t*tile_size + threadIdx.y)*b_cols + col] : 0;
        __syncthreads();
        int dot_prod = 0;
        for (int i = 0; i < tile_size; i++) {
            dot_prod += sharedA[threadIdx.y][i]*sharedB[i][threadIdx.x];
        }
        d_c[row*b_cols + col] += (row < a_rows && col < b_cols) ? dot_prod : 0;

        __syncthreads();
    } 
}

int main() {
    int* a,* b,* c;
    int dim = 100;
    int N = dim* dim;
    size_t bytes = N * sizeof(int);
    a = (int *) malloc(bytes);
    b = (int *) malloc(bytes);
    c = (int *) malloc(bytes);

    for (int i = 0; i < N; i++) {
        a[i] = i % 3;
        b[i] = i % 5;
        c[i] = 0;
    }

    int * d_a, * d_b, * d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    std::cout << "A" << std::endl;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << a[i*dim + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "B" << std::endl;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << b[i*dim + j] << " ";
        }
        std::cout << std::endl;
    }
    dim3 blockDim(tile_size,tile_size);
    dim3 gridDim((dim + tile_size - 1)/tile_size, (dim + tile_size - 1)/tile_size);
    mat_mul<<<gridDim, blockDim>>>(d_a, d_b, d_c, dim, dim, dim);
    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    std::cout << "C" << std::endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << c[i*dim + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
}