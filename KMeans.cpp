#include "helper.h"
#include <cmath>

#define EPS 0.0001

float distance_cpu(
    float* points, 
    long N, 
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

int find_nearest_centroid_cpu(
    float* data, 
    long N, 
    int idx, 
    float* centroids, 
    int k, 
    int n
) {
    int min_index = 0;
    float min_distance = distance_cpu(data, N, idx, centroids, k, 0, n);
    float current_distance;
    for(int i = 1; i < k; i++) {
        current_distance = distance_cpu(data, N, idx, centroids, k, i, n);
        if(current_distance < min_distance) {
            min_distance = current_distance;
            min_index = i;
        } 
    }
    return min_index;
}

void k_means_step_cpu(
    long N, 
    int n, 
    float* data, 
    int k, 
    float* old_centroids, 
    float* new_centroids,
    int* centroid_count,
    int idx
    ) {
    // this function performs one step of the k-means algorithm
    // N - number of data points
    // n - number of features
    // data - pointer to the data array
    // old_centroids - pointer to the old centroids array
    // new_centroids - pointer to the new centroids array

    int min_index = find_nearest_centroid_cpu(data, N, idx, old_centroids, k, n);

    for(int i = 0; i < n; i++) {
        new_centroids[min_index + k * i] += data[idx + N * i];
    }
    centroid_count[min_index] += 1;
}

void generate_label_cpu(
    long N, 
    int n, 
    float* data, 
    int k, 
    float* centroids, 
    int* labels,
    int idx
) {
    // this function generates the labels for the data points
    // N - number of data points
    // n - number of features
    // data - pointer to the data array
    // k - number of centroids
    // centroids - pointer to the centroids array
    // labels - pointer to the labels array
    labels[idx] = find_nearest_centroid_cpu(data, N, idx, centroids, k, n);
}

void divide_cpu(
    float* centroids, 
    int k, 
    int* centroid_count, 
    int n, 
    float* old_centroids, 
    bool* changed,
    int idx
    ) {
    // this function divides the sum of the points by the number of points
    // centroids - pointer to the centroids array
    // k - number of centroids
    // centroid_count - pointer to the centroid count array
    // n - number of features
    if (centroid_count[idx % k] == 0) {
        centroids[idx] = old_centroids[idx];
    } else {
        centroids[idx] /= centroid_count[idx % k]; // idx % k
    }
    if (std::abs(centroids[idx] - old_centroids[idx]) > EPS) {
        *changed = true;
    }
}

void k_means_cpu(long N, int n, float *data, float k, float *centroids, int* iterations, int *labels) {

    float* centroids2 = (float *)malloc(k * n * sizeof(float));
    for(int i = 0; i < k * n; i++) {
        centroids2[i] = 0;
    }
    

    int* centroid_count = (int *)malloc(k * sizeof(int));
    for(int i = 0; i < k; i++) {
        centroid_count[i] = 0;
    }

    bool* changed = new bool;
    *changed = false;


    for(int i = 0; i != *iterations; i++) {
        if (i % 2 == 0) {
            for(int j = 0; j < N; j++) {
                k_means_step_cpu(N, n, data, k, centroids, centroids2, centroid_count, j);
            }
            for(int j = 0; j < k * n; j++) {
                divide_cpu(centroids2, k, centroid_count, n, centroids, changed, j);
            }
        } else {
            for(int j = 0; j < N; j++) {
                k_means_step_cpu(N, n, data, k, centroids2, centroids, centroid_count, j);
            }

            for(int j = 0; j < k * n; j++) {
                divide_cpu(centroids, k, centroid_count, n, centroids2, changed, j);
            }
        }
        if (!*changed) {
            *iterations = i;
            break;

        }
    }
    // depending on the iteration number, the score will be at centroid 1 or 2
    if (*iterations % 2) {
        for(int i = 0; i < k * n; i++) {
            centroids[i] = centroids2[i];
        }
    }

    // generate the labels
    for(int i = 0; i < N; i++) {
        generate_label_cpu(N, n, data, k, centroids, labels, i);
    }

    free(centroids2);
}
