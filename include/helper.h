#ifndef HELPER_H
#define HELPER_H

void display_data(long N, int n, float* data, int* labels);

void generate_data(long N, int n, int n_classes, float *data, int *labels, float spread, float skewness);

void display_data_with_centroids(long N, int n, float* data, int* labels, float* centroids, int k);

void k_means(long N, int n, float *data, float k, float *centroids, int* iterations, int *labels);

void k_means_cpu(long N, int n, float *data, int k, float *centroids, int* iterations, int *labels);

#endif