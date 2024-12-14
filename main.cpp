#include "stdio.h"
#include "helper.h"
#include <stdlib.h>


#define N_DEFINED 2 // for now we need to fix the number of features

int main() {
    long N = 1000000; // numbers of points for each class
    int n = N_DEFINED; // number of features
    int n_classes = 4; // number of classes
    float* data = (float *)malloc(N * n_classes * n * sizeof(float));
    int* labels = (int *)malloc(N * n_classes * sizeof(int));
    float spread = 5;
    float skewness = 0.25;
    
    generate_data(N, n, n_classes, data, labels, spread, skewness);
    
    display_data(N * n_classes, n, data, labels);

    
    int k = 4;
    float centroids[n * k] = {
        1.0, 1.0, -1.0, -1.0,
        1.0, -1.0, 1.0, -1.0
    };
    int* predicted_labels = (int *)malloc(N * n_classes * sizeof(int));
    display_data_with_centroids(N * n_classes, n, data, labels, centroids, k);
    
    int iterations[1] = {100};
    k_means(N * n_classes, n, data, k, centroids, iterations, predicted_labels);

    display_data_with_centroids(N * n_classes, n, data, predicted_labels, centroids, k);
    printf("Iterations: %d\n", iterations[0]);
    
    return 0;
}