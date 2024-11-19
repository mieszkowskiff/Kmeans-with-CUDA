#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define CUDA_CHECK(cudaStatus)                                      \
    if(cudaStatus != cudaSuccess)                                   \
        std::cout << cudaGetErrorString(cudaStatus) << std::endl;   \

__global__ void Add(int* a, int* b, int* c, int n)
{
    //for (int i = 0; i < n; i++)
    //    c[i] = a[i] + b[i];

    int threadID = threadIdx.x;
    int blockID = blockIdx.x;
    int blockDIM = blockDim.x;

    int index = blockID * blockDIM + threadID;

    if (index < n)
        c[index] = a[index] + b[index];
}

int main(int argc, char** argv)
{
    const int N = 10'000;

    int* a = new int[N];
    int* b = new int[N];
    int* c = new int[N];

    for (int i = 0; i < N; i++)
    {
        a[i] = i + 1;
        b[i] = 10 * (i + 1);
        c[i] = 0;
    }

    std::size_t byteCount = N * sizeof(int);

    int* devA; 
    int* devB;
    int* devC;

    CUDA_CHECK(cudaMalloc((void**)&devA, byteCount));
    CUDA_CHECK(cudaMalloc((void**)&devB, byteCount));
    CUDA_CHECK(cudaMalloc((void**)&devC, byteCount));

    CUDA_CHECK(cudaMemcpy(devA, a, byteCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devB, b, byteCount, cudaMemcpyHostToDevice));

    int threads = 1024;
    int blocks = N / threads + 1;

    Add <<<blocks, threads>>> (devA, devB, devC, N);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(c, devC, byteCount, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++)
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;

    delete a;
    delete b;
    delete c;

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
}