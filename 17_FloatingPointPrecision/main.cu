#include <stdio.h>
#include <math.h>

// Kernel for different rounding modes
__global__ void kernel_rn(double* a, double* b, double* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Round to nearest
        result[i] = __dadd_rn(a[i], b[i]);
    }
}

__global__ void kernel_rz(double* a, double* b, double* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Round toward zero
        result[i] = __dadd_rz(a[i], b[i]);
    }
}

__global__ void kernel_ru(double* a, double* b, double* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Round toward positive infinity
        result[i] = __dadd_ru(a[i], b[i]);
    }
}

__global__ void kernel_rd(double* a, double* b, double* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Round toward negative infinity
        result[i] = __dadd_rd(a[i], b[i]);
    }
}

// Function to initialize arrays
void initialize_arrays(double* a, double* b, int n, double large_value, double small_value) {
    for (int i = 0; i < n; i++) {
        a[i] = large_value; // Assign large value
        b[i] = small_value; // Assign small value
    }
}

// Function to compare results
void compare_results(double* rn, double* rz, double* ru, double* rd, int n) {
    printf("Comparing results with different rounding modes:\n");
    for (int i = 0; i < n; i++) {
        printf("Round to nearest (RN): %.10f\n", rn[i]);
        printf("Round toward zero (RZ): %.10f\n", rz[i]);
        printf("Round toward +inf (RU): %.10f\n", ru[i]);
        printf("Round toward -inf (RD): %.10f\n", rd[i]);
        printf("-------------------------------------------------\n");
    }
}

int main() {
    int n = 1;  // Small example with 1 value for simplicity
    double* a, * b, * result_rn, * result_rz, * result_ru, * result_rd;
    double* d_a, * d_b, * d_result_rn, * d_result_rz, * d_result_ru, * d_result_rd;

    // Allocate host memory
    a = (double*)malloc(n * sizeof(double));
    b = (double*)malloc(n * sizeof(double));
    result_rn = (double*)malloc(n * sizeof(double));
    result_rz = (double*)malloc(n * sizeof(double));
    result_ru = (double*)malloc(n * sizeof(double));
    result_rd = (double*)malloc(n * sizeof(double));

    // Initialize the arrays with values that will show rounding differences
    double large_value = 12345678.12345678;   // Large number
    double small_value = 0.00000001;          // Small number with precision difference
    initialize_arrays(a, b, n, large_value, small_value);

    // Allocate device memory
    cudaMalloc(&d_a, n * sizeof(double));
    cudaMalloc(&d_b, n * sizeof(double));
    cudaMalloc(&d_result_rn, n * sizeof(double));
    cudaMalloc(&d_result_rz, n * sizeof(double));
    cudaMalloc(&d_result_ru, n * sizeof(double));
    cudaMalloc(&d_result_rd, n * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_a, a, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernels for each rounding mode
    kernel_rn << <1, n >> > (d_a, d_b, d_result_rn, n);
    kernel_rz << <1, n >> > (d_a, d_b, d_result_rz, n);
    kernel_ru << <1, n >> > (d_a, d_b, d_result_ru, n);
    kernel_rd << <1, n >> > (d_a, d_b, d_result_rd, n);

    // Copy results back to host
    cudaMemcpy(result_rn, d_result_rn, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_rz, d_result_rz, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_ru, d_result_ru, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_rd, d_result_rd, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Compare the results
    compare_results(result_rn, result_rz, result_ru, result_rd, n);

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result_rn);
    cudaFree(d_result_rz);
    cudaFree(d_result_ru);
    cudaFree(d_result_rd);
    free(a);
    free(b);
    free(result_rn);
    free(result_rz);
    free(result_ru);
    free(result_rd);

    return 0;
}
