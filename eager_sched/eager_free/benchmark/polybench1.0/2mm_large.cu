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
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
#include <bemps.hpp>
#include <iostream>
#include <chrono>

// #include <bemps.hpp>

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size. */
# define NI 16384
# define NJ 16384
# define NK 16384
# define NL 16384

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D)
{
	int i, j;

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NK; j++)
		{
			A[i*NI + j] = ((DATA_TYPE) i*j) / NI;
		}
	}

	for (i = 0; i < NK; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			B[i*NK + j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}

	for (i = 0; i < NL; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			C[i*NL + j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{
			D[i*NL + j] = ((DATA_TYPE) i*(j+2)) / NK;	
		}
	}
}


void compareResults(DATA_TYPE *E, DATA_TYPE *E_outputFromGpu)
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < NL; i++)
	{
		for (j=0; j < NI; j++)
		{
			if (percentDiff(E[i*NI + j], E_outputFromGpu[i*NI + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
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


__global__ void mm2_kernel2(DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NL))
	{ 
		int k;
		for (k = 0; k < NJ; k++)
		{
			E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
		}
	}
}


void mm2_cpu(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E)
{
	int i, j, k;
	
  	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			C[i*NJ + j] = 0.0;
			for (k = 0; k < NK; ++k)
			{
				C[i*NJ + j] += A[i*NK + k] * B[k*NJ + j];
			}
		}
	}
	
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{
			E[i*NL + j] = 0.0;
			for (k = 0; k < NJ; ++k)
			{
				E[i*NL + j] += C[i*NJ + k] * D[k*NL + j];
			}
		}
	}
}


void mm2Cuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, DATA_TYPE* E_outputFromGpu, int tid)
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	DATA_TYPE *D_gpu;
	DATA_TYPE *E_gpu;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)ceil( ((float)NJ) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );
	dim3 grid2((size_t)ceil( ((float)NL) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );

	// int64_t membytes = sizeof(DATA_TYPE) * (NI * NK + NK * NJ + NI * NJ + NJ * NL) + 305*1024*1024;
	int64_t membytes = 0;
	membytes += sizeof(DATA_TYPE) * NI * NK;
	membytes += sizeof(DATA_TYPE) * NI * NK;
	membytes += sizeof(DATA_TYPE) * NI * NK;
	membytes += sizeof(DATA_TYPE) * NI * NK;
	membytes += sizeof(DATA_TYPE) * NI * NK;
	membytes += 305 * 1024  * 1024;
	membytes += 20 * 1024 * 1024;

	// printf("membytes: %ld\n", membytes);

	auto bemps_time = std::chrono::high_resolution_clock::now();
	std::cout << "tid: " << tid << ", chrono (bemps): " << bemps_time.time_since_epoch().count() << std::endl;

	// printf("%d: LOG_before\n", tid);
	bemps_begin(tid, grid1.x, grid1.y, grid1.z, block.x, block.y, block.z, membytes);
	// printf("%d: LOG_after\n", tid);

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
	cudaMalloc((void **)&D_gpu, sizeof(DATA_TYPE) * NJ * NL);
	cudaMalloc((void **)&E_gpu, sizeof(DATA_TYPE) * NI * NL);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(D_gpu, D, sizeof(DATA_TYPE) * NJ * NL, cudaMemcpyHostToDevice);
	cudaMemcpy(E_gpu, E, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyHostToDevice);

	// int flag=0;
	// while(1) {
	// 	if (flag) break;
	// 	scanf("%d", &flag);
	// }
		
	

	

	t_start = rtclock();
	mm2_kernel1<<<grid1,block>>>(A_gpu, B_gpu, C_gpu);
	cudaDeviceSynchronize();
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	printf("%d: LOG_before_pre1\n", tid);
	
	pre_bemps_free(tid, (long)sizeof(DATA_TYPE) * ((long)NI * (long)NK + (long)NK * (long)NJ));

	// cudaEventRecord(*pre);
	// cudaEventSynchronize(*pre);


	printf("%d: LOG_after_pre1\n", tid);

	mm2_kernel2<<<grid2,block>>>(C_gpu, D_gpu, E_gpu);
	cudaDeviceSynchronize();
	// cudaFree(C_gpu);
	// cudaFree(D_gpu);
	// printf("%d: LOG_before_pre2\n", tid);
	// pre_bemps_free(tid, sizeof(DATA_TYPE) * (NI * NJ + NJ * NL));
	// pre_bemps_free(tid, (long)sizeof(DATA_TYPE) * ((long)NI * (long)NK + (long)NK * (long)NJ));
	// printf("%d: LOG_after_pre2\n", tid);

	t_end = rtclock();
	// fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	cudaMemcpy(E_outputFromGpu, E_gpu, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyDeviceToHost);

	// cudaFree(A_gpu);
	// cudaFree(B_gpu);
	cudaFree(C_gpu);
	cudaFree(D_gpu);
	cudaFree(E_gpu);

	// printf("%d: LOG_before_end\n", tid);
	bemps_free(tid);
	// cudaEventRecord(*stop);
	// cudaEventSynchronize(*stop);
	// printf("%d: LOG_after_end\n", tid);
}


int main(int argc, char** argv)
{
	// cudaEvent_t start, pre, stop;
	// cudaEventCreate(&start);
	// cudaEventCreate(&pre);
	// cudaEventCreate(&stop);

	// cudaEventRecord(start);

	auto start_time = std::chrono::high_resolution_clock::now();
	int tid = atoi(argv[1]);
	std::cout << "tid: " << tid << ", chrono (start): " << start_time.time_since_epoch().count() << std::endl;

	// dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	// dim3 grid1((size_t)ceil( ((float)NJ) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );
	// dim3 grid2((size_t)ceil( ((float)NL) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );

	// int64_t membytes = 0;
	// membytes += sizeof(DATA_TYPE) * NI * NK;
	// membytes += sizeof(DATA_TYPE) * NI * NK;
	// membytes += sizeof(DATA_TYPE) * NI * NK;
	// membytes += sizeof(DATA_TYPE) * NI * NK;
	// membytes += sizeof(DATA_TYPE) * NI * NK;
	// membytes += 305 * 1024  * 1024;
	// membytes += 20 * 1024 * 1024;

	// bemps_begin(tid, grid1.x, grid1.y, grid1.z, block.x, block.y, block.z, membytes);

	double t_start, t_end;
	
	DATA_TYPE* C;
	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* D;
	DATA_TYPE* E;
	DATA_TYPE* E_outputFromGpu;

	C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));
	D = (DATA_TYPE*)malloc(NJ*NL*sizeof(DATA_TYPE));
	E = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));
	E_outputFromGpu = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));

  	init_array(A, B, C, D);
	// GPU_argv_init();

	mm2Cuda(A, B, C, D, E, E_outputFromGpu, tid);

	// float totalTime, preTime;
	// cudaEventElapsedTime(&totalTime, start, stop);
	// cudaEventElapsedTime(&preTime, start, pre);


	// printf("total time: %f, pre time: %f\n", totalTime, preTime);

	t_start = rtclock();
	// printf("timer starts\n");
	// mm2_cpu(A, B, C, D, E);
	// printf("timer ends\n");
	t_end = rtclock();
	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(E, E_outputFromGpu);

	free(C);
	free(A);
	free(B);
	free(D);
	free(E);
	free(E_outputFromGpu);

	// cudaEventDestroy(start);
    // cudaEventDestroy(pre);
    // cudaEventDestroy(stop);

	// printf("done!\n");
	// bemps_free(tid);

	auto end_time = std::chrono::high_resolution_clock::now();
	std::cout << "tid: " << tid << ", chrono (end): " << start_time.time_since_epoch().count() << std::endl;
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
	std::cout << "tid: " << tid << ", chrono (start to end): " << duration.count() << std::endl;
  	return 0;
}

