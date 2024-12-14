#include "stdio.h"
#include "helper.h"
#include <stdlib.h>
#include <iostream>
#include <chrono>


#define n_DEFINED 2 // number of features must be fixed for data generation

int main() {
    long N = 1000000; // numbers of points for each class
    int n = n_DEFINED; // number of features
    int n_classes = 4; // number of classes
    float* data = (float *)malloc(N * n_classes * n * sizeof(float));
    int* labels = (int *)malloc(N * n_classes * sizeof(int));
    float spread = 5;
    float skewness = 0.25;
    
    auto data_generation_time_start = std::chrono::high_resolution_clock::now();
    generate_data(N, n, n_classes, data, labels, spread, skewness);
    auto data_generation_time_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> data_generation_time_duration = data_generation_time_end - data_generation_time_start;
    std::cout << "Time taken for GPU data generation: " << data_generation_time_duration.count() << " seconds" << std::endl;
    
    display_data(N * n_classes, n, data, labels);

    
    int k = 4;
    float centroids[n * k] = {
        1.0, 1.0, -1.0, -1.0,
        1.0, -1.0, 1.0, -1.0
    };
    int* predicted_labels = (int *)malloc(N * n_classes * sizeof(int));
    display_data_with_centroids(N * n_classes, n, data, labels, centroids, k);
    
    int iterations[1] = {100};

    auto k_means_time_start = std::chrono::high_resolution_clock::now();
    k_means(N * n_classes, n, data, k, centroids, iterations, predicted_labels);
    auto k_means_time_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> k_means_time_duration = k_means_time_end - k_means_time_start;
    std::cout << "Time taken for GPU k-means algorithm: " << k_means_time_duration.count() << " seconds" << std::endl;

    display_data_with_centroids(N * n_classes, n, data, predicted_labels, centroids, k);
    printf("Iterations: %d\n", *iterations);
    
    return 0;
}