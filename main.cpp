#include "stdio.h"
#include "helper.h"


#define N_DEFINED 2 // for now we need to fix the number of features

int main() {
    int N = 100; // numbers of points for each class
    int n = N_DEFINED; // number of features
    int n_classes = 3; // number of classes
    float data[N * n_classes * n];
    int labels[N * n_classes];
    
    generate_data(N, n, n_classes, data, labels);
    /*
    for (int i = 0; i < N * n_classes; i++) {
        for(int j = 0; j < n; j++) printf("%f ", data[i + j * N * n_classes]);
        printf("label: %d \n", labels[i]);
    }
    */
   

    display_data(N * n_classes, n, data, labels);
    return 0;
}