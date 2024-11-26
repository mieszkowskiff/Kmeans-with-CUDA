#include <math.h>
#include <curand_kernel.h>
#include <iostream>

#define N_DEFINED 2

#define CUDA_CHECK(cudaStatus)                                      \
    if(cudaStatus != cudaSuccess)                                   \
        std::cout << cudaGetErrorString(cudaStatus) << std::endl;   \


__device__ float standardNormal(curandState state) {
    float u1 = curand_uniform(&state);
    float u2 = curand_uniform(&state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}


__global__ void generate_data_kernel(int N, int n, int n_classes, float* mi, float* sigma, float* data, int* labels, curandState* states) {
    // this function generates random data for the classification problem
    // N - number of points for each class
    // n - number of features
    // n_classes - number of classes
    // mi - pointer to the mi array
    // sigma - pointer to the sigma array
    // data - pointer to the data array
    // labels - pointer to the labels array

    

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * n_classes) {
        return;
    }

    curand_init(1234, idx, 0, &states[idx]);

    int class_idx = idx / N;
    
    labels[idx] = class_idx;
    

    // generate random data
    float sample[N_DEFINED];
    for (int i = 0; i < n; i++) {
        sample[i] = standardNormal(states[idx]);
    }

    // transform sample to desired distribution
    float transformed_sample[N_DEFINED];
    for (int i = 0; i < n; i++) {
        transformed_sample[i] = mi[class_idx + i * n_classes];
        for (int j = 0; j <= i; j++) {
            transformed_sample[i] += sigma[class_idx + n_classes * (i * (i + 1) / 2 + j)] * sample[j];
        }
        data[idx + i * N * n_classes] = transformed_sample[i];
    }
}




void generate_data(int N, int n, int n_classes, float *data, int *labels) {
    // this function generates random data for the classification problem
    // N - number of points for each class
    // n - number of features
    // n_classes - number of classes
    // data - pointer to the data array
    // labels - pointer to the labels array

    // generate random mi and sigma
    float mi[n_classes * n];
    for (int i = 0; i < n_classes * n; i++) {
        mi[i] = (float)rand() / RAND_MAX;
    }


    // we represent only the lower triangular part of the matrix
    float sigma[n_classes * n * (n + 1) / 2];
    for (int i = 0; i < n_classes * n * (n + 1) / 2; i++) {
        sigma[i] = (float)rand() / RAND_MAX;
    }

    // allocate memory on the device
    int bytes_for_data = N * n_classes * n * sizeof(float);
    float* d_data;

    int bytes_for_labels = N * n_classes * sizeof(int);
    int* d_labels;

    CUDA_CHECK(cudaMalloc((void**)&d_data, bytes_for_data));
    CUDA_CHECK(cudaMalloc((void**)&d_labels, bytes_for_labels));


    // allocate memory for mi and sigma on the device
    float* d_mi;
    float* d_sigma;

    CUDA_CHECK(cudaMalloc((void**)&d_mi, n_classes * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_sigma, n_classes * n * (n + 1) / 2 * sizeof(float)));

    // transfer mi and sigma to the device
    CUDA_CHECK(cudaMemcpy(d_mi, mi, n_classes * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sigma, sigma, n_classes * n * (n + 1) / 2 * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = N * n_classes / threads + 1;


    curandState* d_state;
    CUDA_CHECK(cudaMalloc((void**)&d_state, N * n_classes * sizeof(curandState)));
    // generate data and labels
    generate_data_kernel<<<blocks, threads>>>(N, n, n_classes, d_mi, d_sigma, d_data, d_labels, d_state);

    cudaDeviceSynchronize();
    // copy the data and labels from the device to the host
    CUDA_CHECK(cudaMemcpy(data, d_data, bytes_for_data, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(labels, d_labels, bytes_for_labels, cudaMemcpyDeviceToHost));


    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_mi));
    CUDA_CHECK(cudaFree(d_sigma));
}
