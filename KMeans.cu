#include "helper.h"
#include <iostream>

#define CUDA_CHECK(cudaStatus)                                      \
    if(cudaStatus != cudaSuccess)                                   \
        std::cout << cudaGetErrorString(cudaStatus) << std::endl;   \

__global__ k_means_step(int N, int n, float* data, float* old_centroids, float* new_centroids) {
    // this function performs one step of the k-means algorithm
    // N - number of data points
    // n - number of features
    // data - pointer to the data array
    // old_centroids - pointer to the old centroids array
    // new_centroids - pointer to the new centroids array

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }


    // update the new centroid
    for (int i = 0; i < n; i++) {
        atomicAdd(&new_centroids[closest_centroid * n + i], data[idx * n + i]);
    }
}
void k_means(int N, int n, float* data, float k, float* centroids, int iterations) {
    
    // Allocate memory for the old centroids
    float* d_centroids1;
    CUDA_CHECK(cudaMalloc(&d_centroids1, k * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_centroids1, centroids, k * n * sizeof(float), cudaMemcpyHostToDevice));


    // Allocate memory for the new centroids
    float* d_centroids2;
    CUDA_CHECK(cudaMalloc(&d_centroids2, k * n * sizeof(float)));


    // Allocate memory for the data
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * n * sizeof(float)));

    // Transfer the data to the device
    CUDA_CHECK(cudaMemcpy(d_data, data, N * n * sizeof(float), cudaMemcpyHostToDevice));





    // depending on the iteration number, the score will be at centroid 1 or 2
    if (iterations % 2 == 0) {
        CUDA_CHECK(cudaMemcpy(centroids, d_centroids1, k * n * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy(centroids, d_centroids2, k * n * sizeof(float), cudaMemcpyDeviceToHost));
    }


    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_centroids1));
    CUDA_CHECK(cudaFree(d_centroids2));


}