#ifndef HELPER_H
#define HELPER_H

// Deklaracja funkcji pomocniczej
void display_data(int N, int n, float* data, int* labels);

// Deklaracja funkcji z kernel.cu
void generate_data(int N, int n, int n_classes, float *data, int *labels);

#endif // HELPER_H