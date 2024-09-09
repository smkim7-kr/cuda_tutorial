#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

// The size of the vector
// large data size (bigger than maximum block capacity - 1024 threads)
#define NUM_DATA 1024000

// Simple vector sum kernel (supports larger vectors)
__global__ void vecAdd(int* _a, int* _b, int* _c, int numData) {
    // int tID = threadIdx.x; // this causes an error since different blocks can have same thread index -> redefine thread numbber using thread indexing
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // this resolves the problem of distinguishing same thread index of different blocks
    if (idx < numData) { // when vector_size is not divisible by block_size, this prevents access to unallocated memory
        _c[idx] = _a[idx] + _b[idx];
    }
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
    cudaEvent_t start, stop, copyHtoD_start, kernel_start, copyDtoH_stop;
    float gpu_total_time = 0.0f, copyHtoD_time = 0.0f, kernel_time = 0.0f, copyDtoH_time = 0.0f;

    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&copyHtoD_start);
    cudaEventCreate(&kernel_start);
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

    // 3. Kernel call
    int blockSize = 256; // Number of threads per block
    int gridSize = (NUM_DATA + blockSize - 1) / blockSize; // Number of blocks per grid

    cudaEventRecord(kernel_start, 0);
    vecAdd << <gridSize, blockSize >> > (d_a, d_b, d_c, NUM_DATA); // increase number of blocks since NUM_DATA > 1024 
    

    // 4. Copy results: Device (d_c) -> Host (c)
    cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(copyDtoH_stop, 0);

    // Stop total GPU timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_total_time, start, stop); // Total time
    cudaEventElapsedTime(&copyHtoD_time, start, copyHtoD_start);  // Host to Device copy time
    cudaEventElapsedTime(&kernel_time, copyHtoD_start, kernel_start); // Kernel execution time
    cudaEventElapsedTime(&copyDtoH_time, kernel_start, copyDtoH_stop); // Device to Host copy time

    // Release CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(copyHtoD_start);
    cudaEventDestroy(kernel_start);
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
