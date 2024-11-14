#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void vectorAddWarpSync(int* A, int* B, int* C, int size) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int warpId = tid / warpSize;  // Get the warp ID for this thread

    // Ensure the thread is within bounds of the vector
    if (tid < size) {
        // Step 1: Perform initial addition
        C[tid] = A[tid] + B[tid];

        // Synchronize all threads within the warp
        __syncwarp();

        // Step 2: Imagine some warp-specific processing (like summing within a warp)
        // For simplicity, we multiply the result by the warp ID
        C[tid] *= warpId*2;

        // Synchronize all threads in the warp before moving to the next step
        __syncwarp();

        // Step 3: Final adjustment (just for demonstration)
        C[tid] += warpId;
    }
}

int main() {
    const int size = 64;
    int A[size], B[size], C[size];

    // Initialize vectors A and B
    for (int i = 0; i < size; ++i) {
        A[i] = i;
        B[i] = size - i;
    }

    int* dA, * dB, * dC;
    cudaMalloc(&dA, size * sizeof(int));
    cudaMalloc(&dB, size * sizeof(int));
    cudaMalloc(&dC, size * sizeof(int));

    cudaMemcpy(dA, A, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with 2 blocks and 32 threads each (one warp per block)
    vectorAddWarpSync << <2, 32 >> > (dA, dB, dC, size);

    cudaMemcpy(C, dC, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < size; ++i) {
        printf("C[%d] = %d\n", i, C[i]);
    }

    // Clean up
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
