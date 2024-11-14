#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DATA_TYPE int

#define SIZE_M (512*2)
#define SIZE_N (512*4)
#define SIZE_K (512*2)

#define BLOCK_SIZE 32

// Kernel - shared memory (with bank conflict)
__global__ void MatMul_SharedMem(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    DATA_TYPE val = 0;
    __shared__ DATA_TYPE subA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ DATA_TYPE subB[BLOCK_SIZE][BLOCK_SIZE];

    int localRow = threadIdx.x;
    int localCol = threadIdx.y;

    for (int bID = 0; bID < ceil((float)k / BLOCK_SIZE); bID++) {
        int offset = bID * BLOCK_SIZE;

        // load A and B into shared memory
        if (row >= m || offset + localCol >= k)
            subA[localRow][localCol] = 0;
        else
            subA[localRow][localCol] = matA[row * k + (offset + localCol)];

        if (col >= n || offset + localRow >= k)
            subB[localRow][localCol] = 0;
        else
            subB[localRow][localCol] = matB[(offset + localRow) * n + col];

        __syncthreads();

        // matrix multiplication in shared memory
        for (int i = 0; i < BLOCK_SIZE; i++) {
            val += subA[localRow][i] * subB[i][localCol];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        matC[row * n + col] = val;
    }
}

// Kernel - shared memory (avoiding bank conflict using transposition)
__global__ void MatMul_NoBankConflict(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    DATA_TYPE val = 0;
    __shared__ DATA_TYPE subA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ DATA_TYPE subB[BLOCK_SIZE][BLOCK_SIZE];

    int localRow = threadIdx.x;
    int localCol = threadIdx.y;

    for (int bID = 0; bID < ceil((float)k / BLOCK_SIZE); bID++) {
        int offset = bID * BLOCK_SIZE;

        // load A and B into shared memory with transposition for B to avoid bank conflict
        if (row >= m || offset + localCol >= k)
            subA[localRow][localCol] = 0;
        else
            subA[localRow][localCol] = matA[row * k + (offset + localCol)];

        if (col >= n || offset + localRow >= k)
            subB[localCol][localRow] = 0;  // Transposed to avoid bank conflicts
        else
            subB[localCol][localRow] = matB[(offset + localRow) * n + col];

        __syncthreads();

        // matrix multiplication in shared memory
        for (int i = 0; i < BLOCK_SIZE; i++) {
            val += subA[localRow][i] * subB[localCol][i];  // Transposed access
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        matC[row * n + col] = val;
    }
}

// Basic matrix multiplication kernel (no shared memory)
__global__ void MatMul(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= m || col >= n)
        return;

    DATA_TYPE val = 0;
    for (int i = 0; i < k; i++)
        val += matA[row * k + i] * matB[i * n + col];

    matC[row * n + col] = val;
}

void runMatMul_Basic(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k)
{
    dim3 gridDim(ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    MatMul << <gridDim, blockDim >> > (matA, matB, matC, m, n, k);
    cudaDeviceSynchronize();
}

template<class T> void allocNinitMem(T** p, long long size)
{
    *p = new T[size];
    memset(*p, 0, sizeof(T) * size);
}

int main(int argc, char* argv[])
{
    // set matrix size
    int m, n, k;
    m = SIZE_M;
    n = SIZE_N;
    k = SIZE_K;

    printf("Size : A = (%d by %d), B = (%d by %d), C = (%d by %d)\n", m, k, k, n, m, n);

    int sizeA = m * k;
    int sizeB = k * n;
    int sizeC = m * n;

    // Allocate matrices on host
    DATA_TYPE* A = NULL, * B = NULL;
    allocNinitMem<DATA_TYPE>(&A, sizeA);
    allocNinitMem<DATA_TYPE>(&B, sizeB);

    DATA_TYPE* Ccpu = NULL, * Cgpu = NULL;
    allocNinitMem<DATA_TYPE>(&Ccpu, sizeC);
    allocNinitMem<DATA_TYPE>(&Cgpu, sizeC);

    // Initialize input matrices
    for (int i = 0; i < sizeA; i++) A[i] = rand() % 10;
    for (int i = 0; i < sizeB; i++) B[i] = rand() % 10;

    // Allocate matrices on GPU
    DATA_TYPE* dA, * dB, * dC;
    cudaMalloc(&dA, sizeA * sizeof(DATA_TYPE));
    cudaMalloc(&dB, sizeB * sizeof(DATA_TYPE));
    cudaMalloc(&dC, sizeC * sizeof(DATA_TYPE));

    cudaMemcpy(dA, A, sizeA * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeB * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    // Timing variables using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // Grid and block dimensions
    dim3 gridDim(ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    // No shared memory (basic) kernel timing
    cudaEventRecord(start, 0);
    runMatMul_Basic(dA, dB, dC, m, n, k);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time (basic): %f ms\n", elapsedTime);

    // Shared memory (with bank conflict) kernel timing
    cudaEventRecord(start, 0);
    MatMul_SharedMem << <gridDim, blockDim >> > (dA, dB, dC, m, n, k);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time (shared memory with bank conflict): %f ms\n", elapsedTime);

    // Shared memory (avoiding bank conflict) kernel timing
    cudaEventRecord(start, 0);
    MatMul_NoBankConflict << <gridDim, blockDim >> > (dA, dB, dC, m, n, k);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time (shared memory avoiding bank conflict): %f ms\n", elapsedTime);

    // Clean up
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] A;
    delete[] B;
    delete[] Ccpu;
    delete[] Cgpu;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
