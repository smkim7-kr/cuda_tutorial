
# Lecture: Introduction to GPGPU and CUDA

### Instructor: Duksu Kim, HPC Lab

## Overview

This lecture covers the fundamentals of Graphics Processing Units (GPUs) and General-Purpose computing on GPUs (GPGPU). It introduces parallel computing concepts and the CUDA platform developed by NVIDIA.

## 1. Introduction to GPUs

- GPUs were initially designed as specialized processors for computer graphics.
- The **Graphics Pipeline** is responsible for rendering 2D/3D virtual spaces on the screen.
- Over time, GPUs evolved into powerful processors for general-purpose computing.

## 2. Why GPUs?

- **Parallel Computing**: Many calculations are carried out simultaneously, allowing sub-problems to be solved concurrently.
- **Task Parallelism**: Tasks are distributed across multiple cores (better suited for CPUs).
- **Data Parallelism**: Data is distributed across multiple cores (better suited for GPUs).

## 3. Flynn’s Taxonomy

Classification of computer architectures:

- **SISD**: Single Instruction Single Data
- **SIMD**: Single Instruction Multiple Data (used in many-core GPUs)
- **MISD**: Multiple Instruction Single Data
- **MIMD**: Multiple Instruction Multiple Data

## 4. SIMT (Single Instruction Multiple Threads)

- **SIMT Architecture**: A group of threads (usually 32 threads, called a warp) is controlled by a single control unit.
- Divergent threads in a warp can work with a little performance penalty (work serialization).

## 5. CPU vs GPU

- **CPU**: Optimized for latency and sequential code execution.
  - Focus on the performance of individual cores.
  - Efficient handling of irregular workflows and memory access.
- **GPU**: Optimized for parallelization and throughput.
  - A massive number of cores but less powerful individually.
  - Focus on streaming code and throughput, high performance in floating-point operations.

### CPU Strengths:

- High-performance cores
- Efficient handling of irregular workflows and random memory access patterns.

### GPU Strengths:

- A massive number of cores for parallel tasks.
- Higher performance in terms of FLOPS (Floating-Point Operations per Second).

## 6. Heterogeneous Computing

- Systems that use multiple types of computing resources (CPU + GPU) to solve problems.
- **Advantages**: Fully utilizing available computing resources and achieving higher performance.

## 7. NVIDIA GPUs

- Architectures like Turing, Ampere, etc.
- Used in domains such as gaming (GTX, RTX series), cloud computing (Tesla), and autonomous driving.

## 8. CUDA: Compute Unified Device Architecture

- CUDA is a platform and programming interface designed for utilizing NVIDIA GPUs for parallel computation.
- **Driver API**: Low-level API for more control but harder to program.
- **Runtime API**: High-level API, easier to program but less control.

## CUDA Program Components:

- **Host Code**: Runs on the CPU.
- **Device Code**: Runs on the GPU.
- **Compiler**: NVCC is the compiler for CUDA programs.

## Conclusion

- GPUs are powerful tools for parallel computing.
- The use of both CPU and GPU (heterogeneous computing) can maximize computing performance.
- CUDA provides a flexible platform for GPU programming, allowing developers to harness the power of GPUs effectively.

