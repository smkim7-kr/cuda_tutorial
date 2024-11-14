// main.cpp

#include <iostream>
#include <omp.h>
#include <cuda_runtime.h>
#include "kernel.h"

#define ARRAY_SIZE 1000000

void initializeArray(int* arr, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        arr[i] = i;  // Initialize array with sequential values
    }
}

void processOnCPU(int* arr, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        arr[i] = arr[i] + 1;  // CPU computation
    }
}

int main() {
    int* h_in, * h_out, * d_in, * d_out;

    // Allocate host memory
    h_in = new int[ARRAY_SIZE];
    h_out = new int[ARRAY_SIZE];

    // Initialize input array
    initializeArray(h_in, ARRAY_SIZE);

    // Allocate device memory
    cudaMalloc((void**)&d_in, sizeof(int) * ARRAY_SIZE);
    cudaMalloc((void**)&d_out, sizeof(int) * ARRAY_SIZE);

    // Copy input data from host to device
    cudaMemcpy(d_in, h_in, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    myKernel(d_in, d_out, ARRAY_SIZE);

    // Copy output data from device to host
    cudaMemcpy(h_out, d_out, sizeof(int) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

    // Perform additional computation on the CPU using OpenMP
    processOnCPU(h_out, ARRAY_SIZE);

    // Output some results
    std::cout << "Results: " << h_out[0] << ", " << h_out[ARRAY_SIZE / 2] << ", " << h_out[ARRAY_SIZE - 1] << std::endl;

    // Clean up
    delete[] h_in;
    delete[] h_out;
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
