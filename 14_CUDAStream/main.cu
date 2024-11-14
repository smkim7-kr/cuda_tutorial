#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_BLOCK (128*1024)
#define NUM_T_IN_B 1024
#define ARRAY_SIZE (NUM_T_IN_B * NUM_BLOCK)
#define NUM_STREAMS 4

__global__ void myKernel(int* _in, int* _out)
{
    int tID = blockDim.x * blockIdx.x + threadIdx.x;

    int temp = 0;
    for (int i = 0; i < 250; i++) {
        temp = (temp + _in[tID] * 5) % 10;
    }
    _out[tID] = temp;
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("CUDA Error %s: %s\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}

int main(void)
{
    // Arrays for host and device
    int* in = NULL, * out = NULL;

    // Allocate pinned memory on the host
    checkCudaError(cudaMallocHost(&in, sizeof(int) * ARRAY_SIZE), "cudaMallocHost in");
    checkCudaError(cudaMallocHost(&out, sizeof(int) * ARRAY_SIZE), "cudaMallocHost out");

    // Initialize input array with random values
    for (int i = 0; i < ARRAY_SIZE; i++) {
        in[i] = rand() % 10;
    }

    // Allocate memory on the device
    int* dIn, * dOut;
    checkCudaError(cudaMalloc(&dIn, sizeof(int) * ARRAY_SIZE), "cudaMalloc dIn");
    checkCudaError(cudaMalloc(&dOut, sizeof(int) * ARRAY_SIZE), "cudaMalloc dOut");

    // Create streams
    cudaStream_t stream[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaError(cudaStreamCreate(&stream[i]), "cudaStreamCreate");
    }

    // CUDA event variables for timing for each stream
    cudaEvent_t start[NUM_STREAMS], stop[NUM_STREAMS];
    float elapsedTime[NUM_STREAMS];

    // Create CUDA events for each stream
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaError(cudaEventCreate(&start[i]), "cudaEventCreate start");
        checkCudaError(cudaEventCreate(&stop[i]), "cudaEventCreate stop");
    }

    int chunkSize = ARRAY_SIZE / NUM_STREAMS;

    // Launch async operations with multiple streams and measure time
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = chunkSize * i;

        // Record the start event for this stream
        checkCudaError(cudaEventRecord(start[i], stream[i]), "cudaEventRecord start");

        // Async memory copy and kernel execution
        checkCudaError(cudaMemcpyAsync(dIn + offset, in + offset, sizeof(int) * chunkSize, cudaMemcpyHostToDevice, stream[i]), "cudaMemcpyAsync HtoD");
        myKernel << <NUM_BLOCK / NUM_STREAMS, NUM_T_IN_B, 0, stream[i] >> > (dIn + offset, dOut + offset);
        checkCudaError(cudaMemcpyAsync(out + offset, dOut + offset, sizeof(int) * chunkSize, cudaMemcpyDeviceToHost, stream[i]), "cudaMemcpyAsync DtoH");

        // Record the stop event for this stream
        checkCudaError(cudaEventRecord(stop[i], stream[i]), "cudaEventRecord stop");
    }

    // Synchronize streams and measure elapsed time
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaError(cudaStreamSynchronize(stream[i]), "cudaStreamSynchronize");
        checkCudaError(cudaEventElapsedTime(&elapsedTime[i], start[i], stop[i]), "cudaEventElapsedTime");
    }

    // Print execution times for each stream
    for (int i = 0; i < NUM_STREAMS; i++) {
        printf("Stream %d execution time: %f ms\n", i, elapsedTime[i]);
    }

    // Clean up
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaError(cudaStreamDestroy(stream[i]), "cudaStreamDestroy");
        checkCudaError(cudaEventDestroy(start[i]), "cudaEventDestroy start");
        checkCudaError(cudaEventDestroy(stop[i]), "cudaEventDestroy stop");
    }

    cudaFree(dIn);
    cudaFree(dOut);
    cudaFreeHost(in);
    cudaFreeHost(out);

    return 0;
}
