// colasced memory access improves speed 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define ROW_SIZE (32)
#define K_SIZE   (128)
#define COL_SIZE (32)
#define WORK_LOAD (4096) // access same matrix multiplication multiple times
#define MAT_SIZE_A (ROW_SIZE*K_SIZE)
#define MAT_SIZE_B (K_SIZE*COL_SIZE)
#define MAT_SIZE_C (ROW_SIZE*COL_SIZE)

// Input matrices
float A[ROW_SIZE][K_SIZE];  // m * k
float B[K_SIZE][COL_SIZE];  // k * n

// Kernel for matrix multiplication (standard row and col definition)
__global__ void matMul_kernel(float* _A, float* _B, float* _C) {
    int row = threadIdx.x; // row of A and C
    int col = threadIdx.y; // col of B and C
    int index = row * blockDim.y + col;  // Index in the result matrix

    float result = 0;
    for (int k = 0; k < K_SIZE; k++)
        for (int i = 0; i < WORK_LOAD; i++)
            result += _A[row * K_SIZE + k] * _B[k * COL_SIZE + col];

    _C[index] = result;
}

// Kernel for matrix multiplication with reversed row and col definition
__global__ void matMul_kernel_reversed(float* _A, float* _B, float* _C) {
    int row = threadIdx.y; // reversed row
    int col = threadIdx.x; // reversed col
    int index = row * blockDim.x + col;  // Index in the result matrix

    float result = 0;
    for (int k = 0; k < K_SIZE; k++)
        for (int i = 0; i < WORK_LOAD; i++)
            result += _A[row * K_SIZE + k] * _B[k * COL_SIZE + col];

    _C[index] = result;
}

int main(void) {
    float* dA, * dB, * dC;
    cudaMalloc(&dA, sizeof(float) * MAT_SIZE_A);
    cudaMalloc(&dB, sizeof(float) * MAT_SIZE_B);
    cudaMalloc(&dC, sizeof(float) * MAT_SIZE_C);

    // Initialize input matrices (A and B) with random values
    for (int r = 0; r < ROW_SIZE; r++)
        for (int k = 0; k < K_SIZE; k++)
            A[r][k] = rand() % 100;

    for (int k = 0; k < K_SIZE; k++)
        for (int c = 0; c < COL_SIZE; c++)
            B[k][c] = rand() % 100;

    // Copy input matrices to device
    cudaMemcpy(dA, A, sizeof(float) * MAT_SIZE_A, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * MAT_SIZE_B, cudaMemcpyHostToDevice);

    // Timing variables for CUDA events
    cudaEvent_t start, stop;
    float kernel_time = 0.0f, kernel_reversed_time = 0.0f;

    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockDim(ROW_SIZE, COL_SIZE);

    // Measure time for standard kernel
    cudaEventRecord(start, 0);
    matMul_kernel << <1, blockDim >> > (dA, dB, dC);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);

    // Measure time for reversed kernel
    cudaEventRecord(start, 0);
    matMul_kernel_reversed << <1, blockDim >> > (dA, dB, dC);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_reversed_time, start, stop);

    // Release CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // Print timing results
    printf("Kernel execution time (standard): %f ms\n", kernel_time);
    printf("Kernel execution time (reversed): %f ms\n", kernel_reversed_time);

    return 0;
}
