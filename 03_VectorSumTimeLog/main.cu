#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

// The size of the vector
#define NUM_DATA 1024

// Simple vector sum kernel (Max vector size : 1024)
__global__ void vecAdd(int* _a, int* _b, int* _c) {
    int tID = threadIdx.x;
    _c[tID] = _a[tID] + _b[tID];
}

int main(void)
{
    int* a, * b, * c, * h_c;    // Vectors on the host
    int* d_a, * d_b, * d_c;     // Vectors on the device

    int memSize = sizeof(int) * NUM_DATA;
    printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);

    // Memory allocation on the host-side
    a = new int[NUM_DATA]; memset(a, 0, memSize);
    b = new int[NUM_DATA]; memset(b, 0, memSize);
    c = new int[NUM_DATA]; memset(c, 0, memSize);
    h_c = new int[NUM_DATA]; memset(h_c, 0, memSize);

    // Data generation
    for (int i = 0; i < NUM_DATA; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    // CPU vector sum (for performance comparison)
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_DATA; i++)
        h_c[i] = a[i] + b[i];
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    printf("CPU vector sum time: %f ms\n", cpu_duration.count());

    //****************************************//
    //******* CUDA timing start **************//
    cudaEvent_t start, stop, copyHtoD_start, copyHtoD_stop, kernel_start, kernel_stop, copyDtoH_start, copyDtoH_stop;
    float gpu_total_time = 0.0f, copyHtoD_time = 0.0f, kernel_time = 0.0f, copyDtoH_time = 0.0f;

    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&copyHtoD_start);
    cudaEventCreate(&copyHtoD_stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaEventCreate(&copyDtoH_start);
    cudaEventCreate(&copyDtoH_stop);

    // 1. Memory allocation on the device-side (d_a, d_b, d_c)
    cudaMalloc(&d_a, memSize); cudaMemset(d_a, 0, memSize);
    cudaMalloc(&d_b, memSize); cudaMemset(d_b, 0, memSize);
    cudaMalloc(&d_c, memSize); cudaMemset(d_c, 0, memSize);

    // Start total GPU timing
    cudaEventRecord(start, 0);

    // 2. Data copy: Host (a, b) -> Device (d_a, d_b)
    cudaEventRecord(copyHtoD_start, 0);
    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);
    cudaEventRecord(copyHtoD_stop, 0);
    cudaEventSynchronize(copyHtoD_stop);
    cudaEventElapsedTime(&copyHtoD_time, copyHtoD_start, copyHtoD_stop);

    // 3. Kernel call
    cudaEventRecord(kernel_start, 0);
    vecAdd << <1, NUM_DATA >> > (d_a, d_b, d_c);
    cudaEventRecord(kernel_stop, 0);
    cudaEventSynchronize(kernel_stop);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);

    // 4. Copy results: Device (d_c) -> Host (c)
    cudaEventRecord(copyDtoH_start, 0);
    cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(copyDtoH_stop, 0);
    cudaEventSynchronize(copyDtoH_stop);
    cudaEventElapsedTime(&copyDtoH_time, copyDtoH_start, copyDtoH_stop);

    // Stop total GPU timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_total_time, start, stop);

    // Release CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(copyHtoD_start);
    cudaEventDestroy(copyHtoD_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaEventDestroy(copyDtoH_start);
    cudaEventDestroy(copyDtoH_stop);

    // 5. Release device memory (d_a, d_b, d_c)
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //****************************************//
    //******* CUDA timing end ****************//
    printf("Data copy Host to Device time: %f ms\n", copyHtoD_time);
    printf("Kernel execution time: %f ms\n", kernel_time);
    printf("Data copy Device to Host time: %f ms\n", copyDtoH_time);
    printf("Total GPU run time: %f ms\n", gpu_total_time);

    // Check results
    bool result = true;
    for (int i = 0; i < NUM_DATA; i++) {
        if (h_c[i] != c[i]) {
            printf("[%d] The result is not matched! (%d, %d)\n", i, h_c[i], c[i]);
            result = false;
        }
    }

    if (result)
        printf("GPU works well!\n");

    // Release host memory
    delete[] a; delete[] b; delete[] c; delete[] h_c;

    return 0;
}
