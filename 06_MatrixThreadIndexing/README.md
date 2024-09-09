# Quick Lab 8-4: Thread Indexing in 2D Matrix
This lab introduces how to compute 2D thread indices in CUDA for handling two-dimensional arrays or matrices.

## Key Concepts:
- **Thread Indexing in 2D**: Threads in CUDA can be organized in 2D grids and blocks to map directly to 2D data structures such as matrices.
- **Memory Layout**: Although a matrix is conceptually 2D, it is stored in linear memory, so calculating the correct index in memory requires flattening the 2D grid into a 1D array.

### Steps:
1. **2D Thread Index Calculation**:
   - Each thread in a block can be indexed using `threadIdx.x` for rows and `threadIdx.y` for columns.
   - The final index in the 1D memory space is computed as `index = row * width + column`.
2. **Kernel Design**:
   - The kernel should compute the correct memory address for each matrix element based on the thread and block IDs.

### Concepts Covered:
- Mapping 2D grid and block indices to 1D memory addresses.
- Efficiently handling 2D matrices using CUDA¡¯s parallel execution model.

# Quick Lab 8-5: Determining Grid and Block Size
This lab explains how to choose appropriate grid and block sizes for a CUDA kernel, which directly affects performance.

## Key Concepts:
- **Block Size**: The number of threads in a block.
   - A block should be sized to fully utilize the GPU¡¯s cores while considering the available resources, such as shared memory and registers.
- **Grid Size**: The number of blocks in a grid.
   - The grid size is calculated based on the size of the input data and the number of threads in a block.
- **Limitations**: CUDA hardware imposes limits on the size of blocks and grids, and choosing the correct sizes is essential for optimal performance.

### Steps:
1. **Choosing Block Size**:
   - For optimal performance, each block should have enough threads to maximize GPU utilization (typically 128?1024 threads per block).
2. **Choosing Grid Size**:
   - The grid size is calculated as the total number of data elements divided by the number of threads per block.
3. **Practical Considerations**:
   - The grid and block size should be chosen based on the data size, the GPU's architecture, and the kernel¡¯s resource requirements.

### Concepts Covered:
- Strategies for determining the block and grid size.
- Consideration of GPU resource limits when configuring the execution environment.
- How grid and block sizes impact the performance of CUDA applications.
