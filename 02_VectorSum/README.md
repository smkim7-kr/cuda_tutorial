# Quick Lab 8-2: Vector Sum with CUDA
This lab demonstrates how to perform vector addition using CUDA, illustrating how to use device memory and the GPU for computation.

## Key Concepts:
- **Host vs. Device Memory**: Host memory refers to the CPU's main memory, while device memory refers to the GPU's global memory.
- **Memory Transfer**: `cudaMemcpy()` is used to transfer data between host and device memory.
- **Parallelism in CUDA**: The GPU can process data in parallel by launching multiple threads, each handling a different portion of the data.
- **Kernel Launch Configuration**: The `<<<grid, block>>>` syntax is used to launch a kernel with multiple threads.

### Steps:
1. **Memory Management**:
   - Allocate memory on the host and the device using `cudaMalloc()`.
   - Transfer data from the host to the device using `cudaMemcpy()`.
2. **Kernel Execution**:
   - Define a kernel that performs vector addition, where each thread computes a sum for one element of the vectors.
   - Launch the kernel with a number of threads equal to the size of the vector.
3. **Data Retrieval**:
   - Transfer the results from the device back to the host for verification.

### Concepts Covered:
- Device memory allocation and deallocation.
- Transferring data between CPU and GPU.
- Executing parallel computations on the GPU with multiple threads.
- Kernel execution and memory handling in CUDA.
