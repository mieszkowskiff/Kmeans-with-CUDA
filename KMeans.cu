#include "helper.h"
#include <iostream>

#define CUDA_CHECK(cudaStatus)                                      \
    if(cudaStatus != cudaSuccess)                                   \
        std::cout << cudaGetErrorString(cudaStatus) << std::endl;   \


__device__ float distance(
    float* points, 
    int N, 
    int point_index, 
    float* centroids, 
    int k, 
    int centroid_index, 
    int n
    ){
    // this function calculates the distance between a point and a centroid
    // points - pointer to the points array
    // N - number of points
    // point_index - index of the point
    // centroids - pointer to the centroids array
    // k - number of centroids
    // n - number of features
    float sum = 0;
    for(int i = 0; i < n; i++) {
        sum += (points[point_index + N * i] - centroids[centroid_index + k * i]) * (points[point_index + N * i] - centroids[centroid_index + k * i]);
    }
    return sum;
}

__device__ int find_nearest_centroid(
    float* data, 
    int N, 
    int idx, 
    float* centroids, 
    int k, 
    int n
) {
    int min_index = 0;
    float min_distance = distance(data, N, idx, centroids, k, 0, n);
    float current_distance;
    for(int i = 1; i < k; i++) {
        current_distance = distance(data, N, idx, centroids, k, i, n);
        if(current_distance < min_distance) {
            min_distance = current_distance;
            min_index = i;
        } 
    }
    return min_index;
}

__global__ void k_means_step(
    int N, 
    int n, 
    float* data, 
    int k, 
    float* old_centroids, 
    float* new_centroids,
    int* centroid_count,
    int* labels,
    bool* changed
    ) {
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
    int min_index = find_nearest_centroid(data, N, idx, old_centroids, k, n);

    for(int i = 0; i < n; i++) {
        atomicAdd(&new_centroids[min_index + k * i], data[idx + N * i]);
    }
    atomicAdd(&centroid_count[min_index], 1);
    labels[idx] = min_index;
}

__global__ void divide(float* centroids, int k, int* centroid_count, int n, float* old_centroids) {
    // this function divides the sum of the points by the number of points
    // centroids - pointer to the centroids array
    // k - number of centroids
    // centroid_count - pointer to the centroid count array
    // n - number of features
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * k) {
        return;
    }
    if (centroid_count[idx % k] == 0) {
        centroids[idx] = old_centroids[idx];
    } else {
        centroids[idx] /= centroid_count[idx % k]; // idx % k
    }
    
}

void k_means(int N, int n, float* data, float k, float* centroids, int iterations, int* labels) {
    
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

    int* d_centroid_count;
    CUDA_CHECK(cudaMalloc(&d_centroid_count, k * sizeof(int)));

    int* d_labels;
    CUDA_CHECK(cudaMalloc(&d_labels, N * sizeof(int)));


    bool* d_changed;
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(bool)));


    // number of threads for calculating distances to the centroids
    int iter_threads = 1024;
    int iter_blocks = N / iter_threads + 1;

    // after each iteration, we need to divide the centroids by the number of points
    int divide_threads = 1024;
    int divide_blocks = k * n / divide_threads + 1;

    for(int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemset(d_centroid_count, 0, k * sizeof(int)));
        if (i % 2 == 0) {
            CUDA_CHECK(cudaMemset(d_centroids2, 0, k * n * sizeof(float)));
            cudaDeviceSynchronize();
            k_means_step<<<iter_blocks, iter_threads>>>(
                N, 
                n, 
                d_data, 
                k, 
                d_centroids1, 
                d_centroids2, 
                d_centroid_count, 
                d_labels,
                d_changed
                );
            cudaDeviceSynchronize();
            divide<<<divide_blocks, divide_threads>>>(d_centroids2, k, d_centroid_count, n, d_centroids1);
        } else {
            CUDA_CHECK(cudaMemset(d_centroids1, 0, k * n * sizeof(float)));
            cudaDeviceSynchronize();
            k_means_step<<<iter_blocks, iter_threads>>>(
                N, 
                n, 
                d_data, 
                k, 
                d_centroids2, 
                d_centroids1, 
                d_centroid_count,
                d_labels,
                d_changed
                );
            cudaDeviceSynchronize();
            divide<<<divide_blocks, divide_threads>>>(d_centroids1, k, d_centroid_count, n, d_centroids2);
        }
        cudaDeviceSynchronize();
    }


    // depending on the iteration number, the score will be at centroid 1 or 2
    if (iterations % 2 == 0) {
        CUDA_CHECK(cudaMemcpy(centroids, d_centroids1, k * n * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy(centroids, d_centroids2, k * n * sizeof(float), cudaMemcpyDeviceToHost));
    }

    

    CUDA_CHECK(cudaMemcpy(labels, d_labels, N * sizeof(int), cudaMemcpyDeviceToHost));



    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_centroids1));
    CUDA_CHECK(cudaFree(d_centroids2));
    CUDA_CHECK(cudaFree(d_centroid_count));
    CUDA_CHECK(cudaFree(d_labels));
}