# Quick Lab 8-1: Hello CUDA!
This lab introduces the basics of CUDA programming by printing messages from both the CPU (host) and the GPU (device).

## Key Concepts:
- **CUDA Program Structure**: A CUDA program typically consists of both host code (running on the CPU) and device code (running on the GPU).
- **Kernel**: A CUDA kernel is a function that is executed on the GPU in parallel by multiple threads.
- **Launching a Kernel**: The `<<<...>>>` syntax is used to define the grid and block sizes when launching a kernel from the host.

### Steps:
1. **Set Up a CUDA Project**:
   - Use Visual Studio or any supported IDE.
   - Create a `.cu` file for CUDA source code.
2. **Host-Device Interaction**:
   - Use `__global__` to define a kernel function that runs on the GPU.
   - Print messages from both the CPU and GPU to understand how control is passed between the two.

### Concepts Covered:
- Host and Device code compilation.
- Kernel function execution.
- Setting up a development environment for CUDA with necessary extensions like `.cu` and `.cuh`.
