//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!
//This code is created by Mohammad H Najafi in May 2023

#include "bmm.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// TILEX and TILEY are used to set the number of threads in a CUDA block
#define TILEX 16
#define TILEY 16


// you may define other parameters here!
// you may define other macros here!
// you may define other functions here!

dim3 getDimGrid(const int m, const int n) {
        dim3 dimGrid(n/TILEX,n/TILEY);
        return dimGrid;
}
dim3 getDimBlock(const int m, const int n) {
        dim3 dimBlock(TILEX,TILEY);
        return dimBlock;
}
__global__ void kernelFunc(float* ad, float* bd, float* cd, const int m, const int n) {

        __shared__ float a_sub[TILEY][TILEX];
        __shared__ float b_sub[TILEY][TILEX];

        int row = by * TILEY + ty;
        int col = bx * TILEX + tx;

        float ans_sub = 0.0;
        for (int i = 0; i < n/TILEX; i++) {
                a_sub[ty][tx] = ad[row * n + i * TILEX + tx];
                b_sub[ty][tx] = bd[(i * TILEY + ty) * n + col];
                __syncthreads();

                for(int k = 0 ; k < TILEX ; k++){
                        ans_sub += a_sub[ty][k] * b_sub[k][tx];
                }

		__syncthreads();
        }
        cd[row*n + col] =ans_sub;
}
