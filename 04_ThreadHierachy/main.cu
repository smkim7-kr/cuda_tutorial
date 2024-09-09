 // hierarchical thread layout: grid -> block -> warp -> thread
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void checkIndex(void) {
	printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) gridDim:(%d, %d, %d)\n",
		threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

int main(void) {
	// Total 16 threads
	dim3 dimBlock(2, 2);
	dim3 dimGrid(2, 1, 2);

	printf("dimGrid.x=%d dimGrid.y=%d dimGrid.z=%d\n", dimGrid.x, dimGrid.y, dimGrid.z);
	printf("dimBlock.x=%d dimBlock.y=%d dimBlock.z=%d\n", dimBlock.x, dimBlock.y, dimBlock.z);

	checkIndex << <dimGrid, dimBlock >> > ();

	// Synchronize and clean up device resources
	cudaDeviceReset();
	return 0;
}