// main.cu

#include <cuda_runtime.h>
#include "kernel.h"
#include <stdio.h>

__global__ void cudaKernel(int* d_in, int* d_out, int arraySize)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < arraySize) {
        d_out[idx] = d_in[idx] * 2;  // Simple operation for demonstration
    }
}

void myKernel(int* d_in, int* d_out, int arraySize)
{
    int blockSize = 256;
    int numBlocks = (arraySize + blockSize - 1) / blockSize;

    cudaKernel << <numBlocks, blockSize >> > (d_in, d_out, arraySize);
    cudaDeviceSynchronize();
}
