#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

void checkDeviceMemory(void) {
	size_t free, total;
	cudaMemGetInfo(&free, &total); // build-in function
	printf("Device memory (free/total) = %lld/%lld bytes\n", free, total);
}

int main(void){
	int* dDataPtr; // prefix with d for varirable realted to device (GPU)
	cudaError_t errorCode;

	checkDeviceMemory();
	errorCode = cudaMalloc(&dDataPtr, sizeof(int) * 1024 * 1024); // allocate memory for 1024 * 1024 int type 
	// errorCode = cudaMalloc(&dDataPtr, sizeof(int) * 1024 * 1024 * 1024 * 8); // error when allcocating memory over free available memory - cudaErrorMemoryAllocation
	printf("cudaMalloc - %s\n", cudaGetErrorName(errorCode)); // built-in function: cudaGetErrorName()
	checkDeviceMemory();

	errorCode = cudaMemset(dDataPtr, 0, sizeof(int) * 1024 * 1024); // initialize garbage value to 0
	printf("cudaMemset - %s\n", cudaGetErrorName(errorCode));

	errorCode = cudaFree(dDataPtr); // free memory
	printf("cudaFree - %s\n", cudaGetErrorName(errorCode));
	checkDeviceMemory();
}