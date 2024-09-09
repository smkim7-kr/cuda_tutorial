#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void printData(int* _dDataPtr) {
	printf("%d", _dDataPtr[threadIdx.x]);
}

__global__ void setData(int* _dDataPtr) {
	_dDataPtr[threadIdx.x] = 2;
}

int main(void) {
	int data[10] = { 0 };
	for (int i = 0; i < 10; i++) data[i] = 1; // data array is in host memory (CPU)

	int* dDataPtr;
	cudaMalloc(&dDataPtr, sizeof(int) * 10); // allocate memory for 10 int type size in device memory (GPU)
	cudaMemset(dDataPtr, 0, sizeof(int) * 10);  // initialize all to 0

	printf("Data in device: ");
	printData<<<1, 10>>>(dDataPtr); // call printData() kernel - prints all 0

	cudaMemcpy(dDataPtr, data, sizeof(int) * 10, cudaMemcpyHostToDevice); // copy data array (in CPU) to dDataPtr (in GPU)
	printf("\nHost -> Device: ");
	printData<<<1, 10>>>(dDataPtr); // prints all 1

	setData<<<1, 10>>>(dDataPtr); // set data array in GPU from 1 to 2

	cudaMemcpy(data, dDataPtr, sizeof(int) * 10, cudaMemcpyDeviceToHost); // copy dDataPtr (which has 2) (in GPU) to data (In CPU)
	printf("\nDevice -> Host: ");
	for (int i = 0; i < 10; i++) printf("%d", data[i]); // prints all 2

	cudaFree(dDataPtr); // free memory
}