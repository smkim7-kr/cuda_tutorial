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
    int* in = NULL, * out = NULL, * out2 = NULL;

    // Allocate pinned memory on the host
    checkCudaError(cudaMallocHost(&in, sizeof(int) * ARRAY_SIZE), "cudaMallocHost in");
    checkCudaError(cudaMallocHost(&out, sizeof(int) * ARRAY_SIZE), "cudaMallocHost out");
    checkCudaError(cudaMallocHost(&out2, sizeof(int) * ARRAY_SIZE), "cudaMallocHost out2");

    // Initialize input array with random values
    for (int i = 0; i < ARRAY_SIZE; i++) {
        in[i] = rand() % 10;
    }

    // Allocate memory on the device
    int* dIn, * dOut;
    checkCudaError(cudaMalloc(&dIn, sizeof(int) * ARRAY_SIZE), "cudaMalloc dIn");
    checkCudaError(cudaMalloc(&dOut, sizeof(int) * ARRAY_SIZE), "cudaMalloc dOut");

    // CUDA event variables for timing
    cudaEvent_t start, stop;
    float elapsedTime;

    // Create CUDA events
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

    // ------------------- Single Stream Version -------------------
    printf("Starting Single Stream Version...\n");

    // Start timing
    cudaEventRecord(start, 0);

    // Transfer data from host to device
    checkCudaError(cudaMemcpy(dIn, in, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice), "cudaMemcpy HtoD");

    // Launch the kernel
    myKernel << <NUM_BLOCK, NUM_T_IN_B >> > (dIn, dOut);
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // Transfer data from device to host
    checkCudaError(cudaMemcpy(out, dOut, sizeof(int) * ARRAY_SIZE, cudaMemcpyDeviceToHost), "cudaMemcpy DtoH");

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Single stream execution time: %f ms\n", elapsedTime);

    // ------------------- Multiple Streams Version -------------------
    printf("Starting Multiple Streams Version...\n");

    // Create streams
    cudaStream_t stream[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaError(cudaStreamCreate(&stream[i]), "cudaStreamCreate");
    }

    // Start timing
    cudaEventRecord(start, 0);

    int chunkSize = ARRAY_SIZE / NUM_STREAMS;

    // Launch async operations with multiple streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = chunkSize * i;
        checkCudaError(cudaMemcpyAsync(dIn + offset, in + offset, sizeof(int) * chunkSize, cudaMemcpyHostToDevice, stream[i]), "cudaMemcpyAsync HtoD");
        myKernel << <NUM_BLOCK / NUM_STREAMS, NUM_T_IN_B, 0, stream[i] >> > (dIn + offset, dOut + offset);
        checkCudaError(cudaMemcpyAsync(out2 + offset, dOut + offset, sizeof(int) * chunkSize, cudaMemcpyDeviceToHost, stream[i]), "cudaMemcpyAsync DtoH");
    }

    // Synchronize all streams
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Multiple streams execution time: %f ms\n", elapsedTime);

    // ------------------- Compare Results -------------------
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (out[i] != out2[i]) {
            printf("Mismatch at index %d: Single Stream = %d, Multiple Streams = %d\n", i, out[i], out2[i]);
            break;
        }
    }

    // Clean up
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaError(cudaStreamDestroy(stream[i]), "cudaStreamDestroy");
    }

    cudaFree(dIn);
    cudaFree(dOut);
    cudaFreeHost(in);
    cudaFreeHost(out);
    cudaFreeHost(out2);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
