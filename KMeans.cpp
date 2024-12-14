#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPS 0.0001

float distance_squared(const float *a, const float *b, int n) {
    float dist = 0.0;
    for (int i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// K-means clustering algorithm
void k_means_cpu(long N, int n, float *data, int k, float *centroids, int* iterations, int *labels) {
    int max_iterations = *iterations;
    int changed = 1; // Flag to check if centroids are still moving

    float *new_centroids = (float*)calloc(k * n, sizeof(float)); // To store updated centroids
    int *cluster_sizes = (int*)calloc(k, sizeof(int));           // To count points in each cluster
    float old_centroid_distance;

    for (int iter = 0; iter < max_iterations && changed; iter++) {
        // Reset flags for this iteration
        changed = 0;
        for (int i = 0; i < k * n; i++) new_centroids[i] = 0.0;
        for (int i = 0; i < k; i++) cluster_sizes[i] = 0;

        // Step 1: Assign each point to the nearest centroid
        for (long i = 0; i < N; i++) {
            float min_distance = INFINITY;
            int best_cluster = 0;

            for (int c = 0; c < k; c++) {
                float dist = 0.0;
                for (int d = 0; d < n; d++) {
                    float diff = data[d * N + i] - centroids[d * k + c];
                    dist += diff * diff;
                }
                if (dist < min_distance) {
                    min_distance = dist;
                    best_cluster = c;
                }
            }

            labels[i] = best_cluster; // Assign label to point i

            // Accumulate new centroids
            for (int d = 0; d < n; d++) {
                new_centroids[d * k + best_cluster] += data[d * N + i];
            }
            cluster_sizes[best_cluster]++;
        }

        // Step 2: Update centroids by averaging
        for (int c = 0; c < k; c++) {
            for (int d = 0; d < n; d++) {
                if (cluster_sizes[c] > 0) {
                    new_centroids[d * k + c] /= cluster_sizes[c];
                } else {
                    new_centroids[d * k + c] = centroids[d * k + c]; // Keep the old centroid if no points
                }
            }
        }

        // Step 3: Check if centroids have moved more than EPS
        changed = 0; // Reset changed flag at the start of the check
        for (int c = 0; c < k; c++) {
            float centroid_diff = 0.0;
            for (int d = 0; d < n; d++) {
                float diff = centroids[d * k + c] - new_centroids[d * k + c];
                centroid_diff += diff * diff;
            }
            for (int d = 0; d < n; d++) {
                float diff = fabs(centroids[d * k + c] - new_centroids[d * k + c]);
                if (diff > EPS) {
                    changed = 1; // Mark as changed if any axis moves more than EPS
                }
            }
        }

        // Step 4: Copy new centroids to centroids array
        for (int i = 0; i < k * n; i++) {
            centroids[i] = new_centroids[i];
        }

        if (!changed) {
            *iterations = iter;
            break;
        }
    }

    // Set the total number of iterations performed

    // Cleanup
    free(new_centroids);
    free(cluster_sizes);
}