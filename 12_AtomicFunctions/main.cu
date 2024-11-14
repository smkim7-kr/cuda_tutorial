#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_BLOCK 10240
#define NUM_T_IN_B 512

__global__ void threadCounting_noSync(int* a)
{
    (*a)++;
}

__global__ void threadCounting_atomicGlobal(int* a)
{
    atomicAdd(a, 1);
}

__global__ void threadCounting_atomicShared(int* a)
{
    __shared__ int sa;

    if (threadIdx.x == 0)
        sa = 0;
    __syncthreads();

    atomicAdd(&sa, 1);
    __syncthreads();

    if (threadIdx.x == 0)
        atomicAdd(a, sa);
}

__global__ void threadCounting_warpLvSync(int* a)
{
    __shared__ int wa[NUM_T_IN_B / 32];
    __shared__ int sa;

    int warpID = (int)threadIdx.x / 32;

    if (threadIdx.x % 32 == 0)
        wa[warpID] = 0;
    __syncwarp();

    atomicAdd(&wa[warpID], 1);

    __syncwarp();

    if (threadIdx.x % 32 == 0)
        atomicAdd(&sa, wa[warpID]);

    __syncthreads();

    if (threadIdx.x == 0)
        atomicAdd(a, sa);
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("CUDA Error %s: %s\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}

int main(void) {
    int a = 0;
    int* d1, * d2, * d3, * d4;

    cudaMalloc((void**)&d1, sizeof(int));
    cudaMemset(d1, 0, sizeof(int));

    cudaMalloc((void**)&d2, sizeof(int));
    cudaMemset(d2, 0, sizeof(int));

    cudaMalloc((void**)&d3, sizeof(int));
    cudaMemset(d3, 0, sizeof(int));

    cudaMalloc((void**)&d4, sizeof(int));
    cudaMemset(d4, 0, sizeof(int));

    // CUDA event variables for timing
    cudaEvent_t start, stop;
    float elapsedTime;

    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up kernel
    threadCounting_noSync << <NUM_BLOCK, NUM_T_IN_B >> > (d1);
    cudaDeviceSynchronize();

    // ---------------------- No Sync Kernel ----------------------
    cudaEventRecord(start, 0);
    threadCounting_noSync << <NUM_BLOCK, NUM_T_IN_B >> > (d1);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaMemcpy(&a, d1, sizeof(int), cudaMemcpyDeviceToHost);
    printf("[No Sync.] # of threads = %d\n", a);
    printf("Execution time (No Sync): %f ms\n", elapsedTime);

    // ---------------------- Atomic Global Kernel ----------------------
    cudaEventRecord(start, 0);
    threadCounting_atomicGlobal << <NUM_BLOCK, NUM_T_IN_B >> > (d2);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaMemcpy(&a, d2, sizeof(int), cudaMemcpyDeviceToHost);
    printf("[AtomicGlobal] # of threads = %d\n", a);
    printf("Execution time (Atomic Global): %f ms\n", elapsedTime);

    // ---------------------- Atomic Shared Kernel ----------------------
    cudaEventRecord(start, 0);
    threadCounting_atomicShared << <NUM_BLOCK, NUM_T_IN_B >> > (d3);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaMemcpy(&a, d3, sizeof(int), cudaMemcpyDeviceToHost);
    printf("[AtomicShared] # of threads = %d\n", a);
    printf("Execution time (Atomic Shared): %f ms\n", elapsedTime);

    // ---------------------- Warp Level Sync Kernel ----------------------
    cudaEventRecord(start, 0);
    threadCounting_warpLvSync << <NUM_BLOCK, NUM_T_IN_B >> > (d4);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaMemcpy(&a, d4, sizeof(int), cudaMemcpyDeviceToHost);
    printf("[AtomicWarp] # of threads = %d\n", a);
    printf("Execution time (Atomic Warp): %f ms\n", elapsedTime);

    // Free GPU memory
    cudaFree(d1);
    cudaFree(d2);
    cudaFree(d3);
    cudaFree(d4);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
