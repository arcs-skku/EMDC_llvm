/**
 * 2mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <bitset>
#include <string.h>
using namespace std::chrono;

// #include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size. */
# define NI 8192
# define NJ 8192
# define NK 8192
# define NL 8192

# define page_size 4096
# define VA_block 2097152

# define kernel_num 2

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

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E)
{
	int i, j;

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NK; j++)
		{
			A[i*NI + j] = 1;
		}
	}

	for (i = 0; i < NK; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			B[i*NK + j] = 1;
		}
	}

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			C[i*NI + j] = 0;
		}
	}

	for (i = 0; i < NJ; i++)
	{
		for (j = 0; j < NL; j++)
		{
			D[i*NK + j] = 1;
		}
	}

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{
			E[i*NI + j] = 0;
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

__global__ void mm2_kernel1(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NJ))
	{ 
		int k;
		for (k = 0; k < NK; k++)
		{
			C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
		}
	}
}

int main(int argc, char** argv)
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
	DATA_TYPE* C;
	DATA_TYPE* D;
	DATA_TYPE* E;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)ceil( ((float)NJ) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );

    int per1 = atoi(argv[2]);
	int per2 = atoi(argv[4]);

	CUDA_CHECK(cudaMallocManaged(&A, NI*NK*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&B, NK*NJ*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&C, NI*NJ*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&D, NJ*NL*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&E, NI*NL*sizeof(DATA_TYPE)));

  	init_array(A, B, C, D, E);
	GPU_argv_init();
	
	float tmp = (float)per1 / 100;

	long count1, size1, count2, size2;
	if(strcmp(argv[1], "ZC") == 0){
		count1 = NI * NK * tmp;
		size1 = count1 * 4;
		if(size1 != 0){
			CUDA_CHECK(cudaMemAdvise(A, size1, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
			CUDA_CHECK(cudaMemAdvise(A, size1, cudaMemAdviseSetAccessedBy, 0));
			CUDA_CHECK(cudaMemAdvise(B, size1, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
			CUDA_CHECK(cudaMemAdvise(B, size1, cudaMemAdviseSetAccessedBy, 0));
			CUDA_CHECK(cudaMemAdvise(C, size1, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
			CUDA_CHECK(cudaMemAdvise(C, size1, cudaMemAdviseSetAccessedBy, 0));
			CUDA_CHECK(cudaMemAdvise(D, size1, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
			CUDA_CHECK(cudaMemAdvise(D, size1, cudaMemAdviseSetAccessedBy, 0));
			CUDA_CHECK(cudaMemAdvise(E, size1, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
			CUDA_CHECK(cudaMemAdvise(E, size1, cudaMemAdviseSetAccessedBy, 0));
		}
	}else if(strcmp(argv[1], "GP") == 0){
		count1 = NI * NK * tmp;
		size1 = count1 * 4;
		if(size1 != 0){
			CUDA_CHECK(cudaMemAdvise(A, size1, cudaMemAdviseSetPreferredLocation, 0));
			CUDA_CHECK(cudaMemPrefetchAsync(A, size1, 0, 0));
			CUDA_CHECK(cudaMemAdvise(B, size1, cudaMemAdviseSetPreferredLocation, 0));
			CUDA_CHECK(cudaMemPrefetchAsync(B, size1, 0, 0));
			CUDA_CHECK(cudaMemAdvise(C, size1, cudaMemAdviseSetPreferredLocation, 0));
			CUDA_CHECK(cudaMemPrefetchAsync(C, size1, 0, 0));
			CUDA_CHECK(cudaMemAdvise(D, size1, cudaMemAdviseSetPreferredLocation, 0));
			CUDA_CHECK(cudaMemPrefetchAsync(D, size1, 0, 0));
			CUDA_CHECK(cudaMemAdvise(E, size1, cudaMemAdviseSetPreferredLocation, 0));
			CUDA_CHECK(cudaMemPrefetchAsync(E, size1, 0, 0));
		}
	}else if(strcmp(argv[1], "DP") == 0){
		count1 = NI * NK * tmp;
	}

	count2 = NI * NK - count1;
	if(strcmp(argv[3], "ZC") == 0){
		size2 = count2 * 4;
		if(size2 != 0){
			CUDA_CHECK(cudaMemAdvise((A+count1), size2, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
			CUDA_CHECK(cudaMemAdvise((A+count1), size2, cudaMemAdviseSetAccessedBy, 0));
			CUDA_CHECK(cudaMemAdvise((B+count1), size2, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
			CUDA_CHECK(cudaMemAdvise((B+count1), size2, cudaMemAdviseSetAccessedBy, 0));
			CUDA_CHECK(cudaMemAdvise((C+count1), size2, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
			CUDA_CHECK(cudaMemAdvise((C+count1), size2, cudaMemAdviseSetAccessedBy, 0));
			CUDA_CHECK(cudaMemAdvise((D+count1), size2, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
			CUDA_CHECK(cudaMemAdvise((D+count1), size2, cudaMemAdviseSetAccessedBy, 0));
			CUDA_CHECK(cudaMemAdvise((E+count1), size2, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
			CUDA_CHECK(cudaMemAdvise((E+count1), size2, cudaMemAdviseSetAccessedBy, 0));
		}
	}else if(strcmp(argv[3], "GP") == 0){
		size2 = count2 * 4;
		if(size2 != 0){
			CUDA_CHECK(cudaMemAdvise((A+count1), size2, cudaMemAdviseSetPreferredLocation, 0));
			CUDA_CHECK(cudaMemPrefetchAsync((A+count1), size2, 0, 0));
			CUDA_CHECK(cudaMemAdvise((B+count1), size2, cudaMemAdviseSetPreferredLocation, 0));
			CUDA_CHECK(cudaMemPrefetchAsync((B+count1), size2, 0, 0));
			CUDA_CHECK(cudaMemAdvise((C+count1), size2, cudaMemAdviseSetPreferredLocation, 0));
			CUDA_CHECK(cudaMemPrefetchAsync((C+count1), size2, 0, 0));
			CUDA_CHECK(cudaMemAdvise((D+count1), size2, cudaMemAdviseSetPreferredLocation, 0));
			CUDA_CHECK(cudaMemPrefetchAsync((D+count1), size2, 0, 0));
			CUDA_CHECK(cudaMemAdvise((E+count1), size2, cudaMemAdviseSetPreferredLocation, 0));
			CUDA_CHECK(cudaMemPrefetchAsync((E+count1), size2, 0, 0));
		}
	}else if(strcmp(argv[1], "DP") == 0){

	}

	mm2_kernel1<<<grid1, block>>>(A, B, C);
	CUDA_CHECK(cudaDeviceSynchronize());

	mm2_kernel1<<<grid1,block>>>(C, D, E);
	CUDA_CHECK(cudaDeviceSynchronize());

	for (int i = 0; i < NI; i++)
	{
		for (int j = 0; j < NL; j++)
		{
			if(E[i*NI + j] != NI*NL){
				printf("Err\n");
				break;
			}
		}
	}

	CUDA_CHECK(cudaFree(A));
	CUDA_CHECK(cudaFree(B));
	CUDA_CHECK(cudaFree(C));
	CUDA_CHECK(cudaFree(D));
	CUDA_CHECK(cudaFree(E));

	clock_gettime( CLOCK_REALTIME, &specific_time);
    now = localtime(&specific_time.tv_sec);
    millsec = specific_time.tv_nsec;

    millsec = floor (specific_time.tv_nsec/1.0e6);

    printf("Finish, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", 1900 + now->tm_year, 
        now->tm_mon + 1, now->tm_mday, now->tm_hour, 
        now->tm_min, now->tm_sec, millsec);

  	return 0;
}
