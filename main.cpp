#include "stdio.h"
#include "helper.h"


#define N_DEFINED 2 // for now we need to fix the number of features

int main() {
    int N = 10; // numbers of points for each class
    int n = N_DEFINED; // number of features
    int n_classes = 10; // number of classes
    float data[N * n_classes * n];
    int labels[N * n_classes];
    
    generate_data(N, n, n_classes, data, labels);
    /*
    for (int i = 0; i < N * n_classes; i++) {
        for(int j = 0; j < n; j++) printf("%f ", data[i + j * N * n_classes]);
        printf("label: %d \n", labels[i]);
    }
    */
    // display_data(N * n_classes, n, data, labels);


    int k = 4;
    float centroids[n * k] = {
        1.0, 1.0, -1.0, -1.0,
        1.0, -1.0, 1.0, -1.0
    };
    int predicted_labels[N * n_classes];
    display_data_with_centroids(N * n_classes, n, data, labels, centroids, k);
    
    int iterations[1] = {100000};
    k_means(N * n_classes, n, data, k, centroids, iterations, predicted_labels);

    display_data_with_centroids(N * n_classes, n, data, predicted_labels, centroids, k);
    printf("Iterations: %d\n", iterations[0]);
    return 0;
}