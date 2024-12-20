#include "stdio.h"
#include "helper.h"
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <memory>

#define n_DEFINED 2 // number of features must be fixed for data generation

int main() {
    long N = 1000000; // numbers of points for each class
    int n = n_DEFINED; // number of features 
    int n_classes = 4; // number of classes (20 at most, because of the color palette)
    float* data = (float *)malloc(N * n_classes * n * sizeof(float));
    int* labels = (int *)malloc(N * n_classes * sizeof(int));

    // parameters for data generation
    float spread = 5;
    float skewness = 0.25;
    
    auto data_generation_time_start = std::chrono::high_resolution_clock::now();
    generate_data(N, n, n_classes, data, labels, spread, skewness);
    auto data_generation_time_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> data_generation_time_duration = data_generation_time_end - data_generation_time_start;
    std::cout << "Time taken for GPU data generation: " << data_generation_time_duration.count() << " seconds" << std::endl;
    
    display_data(N * n_classes, n, data, labels);

    
    int k = 4;
    float* centroids = new float[n * k];

    // Centroids initialization
    centroids[0] = 1.0;
    centroids[1] = 1.0;
    centroids[2] = -1.0;
    centroids[3] = -1.0;
    centroids[4] = 1.0;
    centroids[5] = -1.0;
    centroids[6] = 1.0;
    centroids[7] = -1.0;

    int* predicted_labels = (int *)malloc(N * n_classes * sizeof(int));
    display_data_with_centroids(N * n_classes, n, data, labels, centroids, k);
    
    int iterations[1] = {1000};

    auto k_means_time_start_gpu = std::chrono::high_resolution_clock::now();
    k_means(N * n_classes, n, data, k, centroids, iterations, predicted_labels);
    auto k_means_time_end_gpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> k_means_time_duration_gpu = k_means_time_end_gpu - k_means_time_start_gpu;
    std::cout << "Time taken for GPU k-means algorithm: " << k_means_time_duration_gpu.count() << " seconds" << std::endl;

    display_data_with_centroids(N * n_classes, n, data, predicted_labels, centroids, k);
    printf("Iterations: %d\n", *iterations);


    // cpu version
    centroids[0] = 1.0;
    centroids[1] = 1.0;
    centroids[2] = -1.0;
    centroids[3] = -1.0;
    centroids[4] = 1.0;
    centroids[5] = -1.0;
    centroids[6] = 1.0;
    centroids[7] = -1.0;


    display_data_with_centroids(N * n_classes, n, data, labels, centroids, k);
    
    iterations[0] = 1000;

    
    auto k_means_time_start_cpu = std::chrono::high_resolution_clock::now();
    
    k_means_cpu(N * n_classes, n, data, k, centroids, iterations, predicted_labels);
    
    auto k_means_time_end_cpu = std::chrono::high_resolution_clock::now();


    std::chrono::duration<double> k_means_time_duration_cpu = k_means_time_end_cpu - k_means_time_start_cpu;
    std::cout << "Time taken for CPU k-means algorithm: " << k_means_time_duration_cpu.count() << " seconds" << std::endl;


    display_data_with_centroids(N * n_classes, n, data, predicted_labels, centroids, k);
    printf("Iterations: %d\n", *iterations);

    free(data);
    free(labels);
    free(predicted_labels);
    delete[] centroids;


    return 0;
}


int create_data_for_experiments() {
    int n = n_DEFINED; // number of features
    int n_classes = 10; // number of classes
    float* data;
    int* labels;
    float spread = 5;
    float skewness = 0.2;
    int k = 4;
    int iterations[1] = {1000};
    float* centroids = new float[n * k];
    std::chrono::duration<double> k_means_time_duration_gpu;
    std::chrono::duration<double> k_means_time_duration_cpu;
    std::chrono::duration<double> data_creation_time_duration;

    auto k_means_time_start_gpu = std::chrono::high_resolution_clock::now();
    auto k_means_time_end_gpu = std::chrono::high_resolution_clock::now();
    auto k_means_time_start_cpu = std::chrono::high_resolution_clock::now();
    auto k_means_time_end_cpu = std::chrono::high_resolution_clock::now();
    auto data_creation_time_start = std::chrono::high_resolution_clock::now();
    auto data_creation_time_end = std::chrono::high_resolution_clock::now();


    std::ofstream outFile; // Create an output file stream object
    const std::string filename = "result.txt";

    if (!outFile) {
        std::cerr << "Error: File could not be opened!" << std::endl;
        return 1;
    }

    // Open the file in write mode
    outFile.open(filename);

    // Change these values to get the desired range of N
    for(int N = 228000; N < 1000000; N = N + 1000) {
        iterations[0] = 1000;

        centroids[0] = 1.0;
        centroids[1] = 1.0;
        centroids[2] = -1.0;
        centroids[3] = -1.0;
        centroids[4] = 1.0;
        centroids[5] = -1.0;
        centroids[6] = 1.0;
        centroids[7] = -1.0;

        data = (float *)malloc(N * n_classes * n * sizeof(float));
        labels = (int *)malloc(N * n_classes * sizeof(int));

        data_creation_time_start = std::chrono::high_resolution_clock::now();
        generate_data(N, n, n_classes, data, labels, spread, skewness);
        data_creation_time_end = std::chrono::high_resolution_clock::now();

        data_creation_time_duration = data_creation_time_end - data_creation_time_start;

        k_means_time_start_gpu = std::chrono::high_resolution_clock::now();
        k_means(N * n_classes, n, data, k, centroids, iterations, labels);
        k_means_time_end_gpu = std::chrono::high_resolution_clock::now();

        k_means_time_duration_gpu = k_means_time_end_gpu - k_means_time_start_gpu;

        iterations[0] = 1000;

        centroids[0] = 1.0;
        centroids[1] = 1.0;
        centroids[2] = -1.0;
        centroids[3] = -1.0;
        centroids[4] = 1.0;
        centroids[5] = -1.0;
        centroids[6] = 1.0;
        centroids[7] = -1.0;

        k_means_time_start_cpu = std::chrono::high_resolution_clock::now();
        k_means_cpu(N * n_classes, n, data, k, centroids, iterations, labels);
        k_means_time_end_cpu = std::chrono::high_resolution_clock::now();

        k_means_time_duration_cpu = k_means_time_end_cpu - k_means_time_start_cpu;

        free(data);
        free(labels);
        outFile << N << ";" << data_creation_time_duration.count() << ";" << k_means_time_duration_gpu.count() << ";" << k_means_time_duration_cpu.count() << ";" << *iterations << std::endl;

    }
    outFile.flush();
    outFile.close();

    return 0;
}
