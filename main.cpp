#include "stdio.h"
#include "helper.h"


#define N_DEFINED 2 // for now we need to fix the number of features

int main() {
    int N = 10; // numbers of points for each class
    int n = N_DEFINED; // number of features
    int n_classes = 4; // number of classes
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


    int k = 2;
    float centroids[n * k] = {
        0.0, 0.0,
        -1.0, 1.0
    };

    display_data_with_centroids(N * n_classes, n, data, labels, centroids, k);
    

    k_means(N * n_classes, n, data, k, centroids, 1);

    display_data_with_centroids(N * n_classes, n, data, labels, centroids, k);

    for (int i = 0; i < k; i++) {
        printf("Centroid %d: %f %f\n", i, centroids[i], centroids[i + k]);
    }
    


    return 0;
}