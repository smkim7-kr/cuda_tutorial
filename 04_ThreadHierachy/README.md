# Quick Lab 8-3: Understanding Thread Layout
This lab helps understand how threads are organized in CUDA and how the grid, block, and thread hierarchy is constructed.

## Key Concepts:
- **CUDA Thread Hierarchy**:
   - **Thread**: The smallest unit of execution.
   - **Block**: A group of threads that can communicate and share memory.
   - **Grid**: A collection of blocks that can execute a kernel.
- **Thread and Block IDs**: Each thread and block has a unique ID, which is used to compute the work they perform.

### Steps:
1. **Thread Indexing**:
   - Each thread is assigned a unique index using `threadIdx` for threads within a block and `blockIdx` for blocks within a grid.
2. **Configuring Execution**:
   - Define the number of blocks and threads per block using the `dim3` data type for 1D, 2D, or 3D configurations.
   - Use `blockIdx` and `threadIdx` to identify the work assigned to each thread.

### Concepts Covered:
- The structure of CUDA's execution model.
- The role of thread, block, and grid dimensions in organizing parallel work.
- How thread and block IDs are used to divide tasks in a parallel program.
