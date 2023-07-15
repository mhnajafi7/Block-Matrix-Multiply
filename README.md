# Block-Matrix-Multiply
This repository provides an implementation of block matrix multiplication using tiled matrix multiplication algorithm on NVIDIA GPUs. The tiled matrix multiplication algorithm is an optimization technique that partitions the matrices into smaller submatrices, or tiles, and computes the matrix multiplication of these tiles. This approach reduces the number of global memory transactions and allows for efficient memory access, resulting in faster computation times. The code is designed to handle large matrices and takes advantage of the parallel processing capabilities of NVIDIA GPUs to significantly accelerate matrix multiplication.

The code consists of the following files:

1. `bmm.h`: Header file containing the function prototypes for the tiled matrix multiplication algorithm.
2. `bmm.cu`: This file contains the CUDA implementation of the tiled matrix multiplication algorithm, which is used to perform block matrix multiplication using shared memory and CUDA parallelism.
3. `bmm_main.cu`: This file is the entry point for running the block matrix multiplication using the tiled matrix multiplication algorithm. It contains the main function for calling the bmm function, which performs the matrix multiplication using the kernelFunc function from bmm.cu.
4. `gpuerrors.h`: Header file containing functions for checking and reporting CUDA errors.
5. `gputimer.h`: Header file containing functions for measuring GPU execution time.

## Requirements
To run the code, you will need:
* **NVIDIA** GPU with compute capability of at least 2.0
* **CUDA** toolkit installed (version 7.5 or later)
* **C++** compiler (supporting C++11 standard)

## Usage 
To compile all of the code, you can run the following command after saving them:
```bash
    nvcc -O2 bmm_main.cu bmm.cu -o bmm
```
To run the compiled code, you can enter the appropriate command (10 <= M <= 13):
```bash
    ./bmm M
```
## Implementation Details
The tiled matrix multiplication algorithm works by partitioning the matrices into smaller submatrices, or tiles, and computing the matrix multiplication of these tiles. The size of the tiles is determined by the TILEX and TILEY macros defined in the `bmm.cu` file. The algorithm uses shared memory to store the tiles, which allows for efficient memory access and reduces the number of global memory transactions.

The *getDimGrid* and *getDimBlock* functions are used to calculate the dimensions of the CUDA grid and block, respectively. The *kernelFunc* function is the main CUDA kernel that performs the tiled matrix multiplication. It uses shared memory to load the tiles into the GPU, and then performs the matrix multiplication using the tiles.
