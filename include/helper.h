#ifndef HELPER_H
#define HELPER_H

//declaration of display_data
void display_data(int N, int n, float* data, int* labels);

// Declaration of kernel wrapper
void generate_data(int N, int n, int n_classes, float *data, int *labels);

void k_means(int N, int n, float *data, float k, float *centroids, int iterations = 100);

#endif // HELPER_H