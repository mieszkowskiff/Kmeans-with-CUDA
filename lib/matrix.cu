#include <stdio.h>
#include <iostream>

#define CUDA_CHECK(cudaStatus)                                      \
    if(cudaStatus != cudaSuccess)                                   \
        std::cout << cudaGetErrorString(cudaStatus) << std::endl;   \




__global__ void multiply(int N, int* x, int *A, int* y) {
    int threadID = threadIdx.x;
    int blockID = blockIdx.x;
    int blockDIM = blockDim.x;

    y[threadID] = 0;
    for (int i = 0; i < N; i++)
        y[threadID] += A[threadID * N + i] * x[i];
}



int main(int argc, char** argv)
{
    const int N = 2;

    int* x = new int[N];
    int* A = new int[N * N];

    x[0] = 1;
    x[1] = 2;

    A[0] = 1;
    A[1] = 2;
    A[2] = 3;
    A[3] = 4;

    int* d_x;
    int* d_A;

    size_t byteCount_x = N * sizeof(int);
    size_t byteCount_A = N * N * sizeof(int);
    CUDA_CHECK(cudaMalloc((void**)&d_x, byteCount_x));
    CUDA_CHECK(cudaMalloc((void**)&d_A, byteCount_A));
    CUDA_CHECK(cudaMemcpy(d_x, x, byteCount_x, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, A, byteCount_A, cudaMemcpyHostToDevice));

    int* y = new int[N];
    int* d_y;

    CUDA_CHECK(cudaMalloc((void**)&d_y, byteCount_x));
    int threads = N;
    int blocks = 1;

    multiply <<<blocks, threads>>>(N, d_x, d_A, d_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(y, d_y, byteCount_x, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++)
    {
        std::cout << "y[" << i << "] = " << y[i] << std::endl;
    }
}