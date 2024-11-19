#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H

#include <curand_kernel.h>
#include <cuda_runtime.h>


void generate_data(int N, int n, int n_classes, float *data, int *labels);

__global__ void generate_data_kernel(int N, int n, int n_classes, float* mi, float* sigma, float* data, int* labels, curandState* states);

__device__ float standardNormal(curandState states);

#endif