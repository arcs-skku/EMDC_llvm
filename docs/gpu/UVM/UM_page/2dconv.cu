/**
 * 2DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <bitset>
#include <string.h>
using namespace std::chrono;
 
//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05
 
#define GPU_DEVICE 0

# define page_size 4096
# define VA_block 2097152

# define kernel_num 1

/* Problem size */
#define NI 8192
#define NJ 8192
 
/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 32
 
 /* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

#define CUDA_CHECK(val) { \
	if (val != cudaSuccess) { \
		fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(val), __LINE__, __FILE__); \
		exit(val); \
	} \
}

void init(DATA_TYPE* A, DATA_TYPE* B)
{
    int i, j;
 
    for (i = 0; i < NI; ++i)
    {
        for (j = 0; j < NJ; ++j)
        {
            // A[i*NJ + j] = (float)rand()/RAND_MAX;
            A[i*NJ + j] = 1;
            B[i*NJ + j] = 0;
        }
     }
}
 
 
 void GPU_argv_init()
 {
     cudaDeviceProp deviceProp;
     cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
     printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
     cudaSetDevice( GPU_DEVICE );
 }
 
 
__global__ void Convolution2D_kernel(DATA_TYPE *A, DATA_TYPE *B)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
 
    DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;
 
    c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
    c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
    c13 = +0.4;  c23 = +0.7;  c33 = +0.10;
 
    if ((i < NI-1) && (j < NJ-1) && (i > 0) && (j > 0))
    {
        B[i * NJ + j] =  c11 * A[(i - 1) * NJ + (j - 1)]  + c21 * A[(i - 1) * NJ + (j + 0)] + c31 * A[(i - 1) * NJ + (j + 1)] 
            + c12 * A[(i + 0) * NJ + (j - 1)]  + c22 * A[(i + 0) * NJ + (j + 0)] +  c32 * A[(i + 0) * NJ + (j + 1)]
            + c13 * A[(i + 1) * NJ + (j - 1)]  + c23 * A[(i + 1) * NJ + (j + 0)] +  c33 * A[(i + 1) * NJ + (j + 1)];
    }
}

int main(int argc, char *argv[])
{
    cudaFree(0);
 
	struct timespec specific_time;
    struct tm *now;
    int millsec;
    clock_gettime( CLOCK_REALTIME, &specific_time);
    now = localtime(&specific_time.tv_sec);
    millsec = specific_time.tv_nsec;

    millsec = floor (specific_time.tv_nsec/1.0e6);

    printf("Start, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", 1900 + now->tm_year, 
        now->tm_mon + 1, now->tm_mday, now->tm_hour, 
        now->tm_min, now->tm_sec, millsec);

    DATA_TYPE* A;
    DATA_TYPE* B;

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((size_t)ceil( ((float)NI) / ((float)block.x) ), (size_t)ceil( ((float)NJ) / ((float)block.y)) );

    int per1 = atoi(argv[2]);
	int per2 = atoi(argv[4]);

    CUDA_CHECK(cudaMallocManaged(&A, NI*NJ*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&B, NI*NJ*sizeof(DATA_TYPE)));
 
    init(A, B);
    GPU_argv_init();

	float tmp = (float)per1 / 100;

	long count1, size1, count2, size2;
	if(strcmp(argv[1], "ZC") == 0){
		count1 = NI * NJ * tmp;
		size1 = count1 * 4;
		if(size1 != 0){
			CUDA_CHECK(cudaMemAdvise(A, size1, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
			CUDA_CHECK(cudaMemAdvise(A, size1, cudaMemAdviseSetAccessedBy, 0));
			CUDA_CHECK(cudaMemAdvise(B, size1, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
			CUDA_CHECK(cudaMemAdvise(B, size1, cudaMemAdviseSetAccessedBy, 0));
		}
	}else if(strcmp(argv[1], "GP") == 0){
		count1 = NI * NJ * tmp;
		size1 = count1 * 4;
		if(size1 != 0){
			CUDA_CHECK(cudaMemAdvise(A, size1, cudaMemAdviseSetPreferredLocation, 0));
			CUDA_CHECK(cudaMemPrefetchAsync(A, size1, 0, 0));
			CUDA_CHECK(cudaMemAdvise(B, size1, cudaMemAdviseSetPreferredLocation, 0));
			CUDA_CHECK(cudaMemPrefetchAsync(B, size1, 0, 0));
		}
	}else if(strcmp(argv[1], "DP") == 0){
		count1 = NI * NJ * tmp;
	}

	count2 = NI * NJ - count1;
	if(strcmp(argv[3], "ZC") == 0){
		size2 = count2 * 4;
		if(size2 != 0){
			CUDA_CHECK(cudaMemAdvise((A+count1), size2, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
			CUDA_CHECK(cudaMemAdvise((A+count1), size2, cudaMemAdviseSetAccessedBy, 0));
			CUDA_CHECK(cudaMemAdvise((B+count1), size2, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
			CUDA_CHECK(cudaMemAdvise((B+count1), size2, cudaMemAdviseSetAccessedBy, 0));
		}
	}else if(strcmp(argv[3], "GP") == 0){
		size2 = count2 * 4;
		if(size2 != 0){
			CUDA_CHECK(cudaMemAdvise((A+count1), size2, cudaMemAdviseSetPreferredLocation, 0));
			CUDA_CHECK(cudaMemPrefetchAsync((A+count1), size2, 0, 0));
			CUDA_CHECK(cudaMemAdvise((B+count1), size2, cudaMemAdviseSetPreferredLocation, 0));
			CUDA_CHECK(cudaMemPrefetchAsync((B+count1), size2, 0, 0));
		}
	}else if(strcmp(argv[1], "DP") == 0){

	}

    Convolution2D_kernel<<<grid,block>>>(A, B);
    CUDA_CHECK(cudaDeviceSynchronize());

	for (int i = 0; i < NI; i++)
	{
		for (int j = 0; j < NJ; j++)
		{
			B[i*NI+j] += 1;
		}
	}

    printf("Check val: %f\n", B[0]);
     
    cudaFree(A);
    cudaFree(B);

    clock_gettime( CLOCK_REALTIME, &specific_time);
    now = localtime(&specific_time.tv_sec);
    millsec = specific_time.tv_nsec;

    millsec = floor (specific_time.tv_nsec/1.0e6);

    printf("Finish, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", 1900 + now->tm_year, 
        now->tm_mon + 1, now->tm_mday, now->tm_hour, 
        now->tm_min, now->tm_sec, millsec);
    
    return 0;
}
 
 
