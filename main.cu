#include "stdio.h"
#include "lib/data_generator.h"


#define N_DEFINED 1

int main() {
    int N = 4; // numbers of points for each class
    int n = N_DEFINED; // number of features
    int n_classes = 2; // number of classes
    float data[N * n_classes * n];
    int labels[N * n_classes];
    generate_data(N, n, n_classes, data, labels);
    for (int i = 0; i < N * n_classes * n; i++) {
        printf("%f ", data[i]);
    }

    return 0;
}