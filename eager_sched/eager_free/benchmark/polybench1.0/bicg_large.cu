/**
 * bicg.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <sys/time.h>
#include <cuda.h>
#include <bemps.hpp>
#include <iostream>
#include <chrono>

#include "../../common/polybenchUtilFuncts.h"

//Error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size. */
#define NX 32768
#define NY 32768

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_array(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *r)
{
	int i, j;

  	for (i = 0; i < NX; i++)
	{
    		r[i] = i * M_PI;

    		for (j = 0; j < NY; j++)
		{
      			A[i*NY + j] = ((DATA_TYPE) i*j) / NX;
		}
 	}
	
	for (i = 0; i < NY; i++)
	{
    		p[i] = i * M_PI;
	}
}


void compareResults(DATA_TYPE* s, DATA_TYPE* s_outputFromGpu, DATA_TYPE* q, DATA_TYPE* q_outputFromGpu)
{
	int i,fail;
	fail = 0;

	// Compare s with s_cuda
	for (i=0; i<NX; i++)
	{
		if (percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}

	for (i=0; i<NY; i++)
	{
		if (percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
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


//Distributed (split) from initial loop and permuted into reverse order to allow parallelism...
__global__ void bicg_kernel1(DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < NY)
	{
		s[j] = 0.0f;

		int i;
		for(i = 0; i < NX; i++)
		{
			s[j] += A[i * NY + j] * r[i];
		}
	}	
}


//Distributed (split) from initial loop to allow parallelism
__global__ void bicg_kernel2(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *q)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < NX)
	{
		q[i] = 0.0f;

		int j;
		for(j=0; j < NY; j++)
		{
			q[i] += A[i * NY + j] * p[j];
		}
	}
}


void bicg_cpu(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q)
{
	int i,j;
	
  	for (i = 0; i < NY; i++)
	{
		s[i] = 0.0;
	}

    for (i = 0; i < NX; i++)
    {
		q[i] = 0.0;
		for (j = 0; j < NY; j++)
	  	{
	    		s[j] = s[j] + r[i] * A[i*NY + j];
	    		q[i] = q[i] + A[i*NY + j] * p[j];
	  	}
	}
}


void bicgCuda(int tid, DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q,
			DATA_TYPE* s_outputFromGpu, DATA_TYPE* q_outputFromGpu)
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *q_gpu;
	DATA_TYPE *p_gpu;
	DATA_TYPE *r_gpu;
	DATA_TYPE *s_gpu;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);

	int64_t membytes = sizeof(DATA_TYPE) * (NX * NY + NX + NY + NY + NX);
	membytes += 305 * 1024 * 1024;
	membytes += 20 * 1024 * 1024;
	printf("membytes: %ld\n", membytes);

	auto bemps_time = std::chrono::high_resolution_clock::now();
	std::cout << "tid: " << tid << ", chrono (bemps): " << bemps_time.time_since_epoch().count() << std::endl;

	bemps_begin(tid, max(grid1.x, grid2.x), max(grid1.y, grid2.y), max(grid1.z, grid2.z), block.x, block.y, block.z, membytes);

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NX * NY);
	cudaMalloc((void **)&r_gpu, sizeof(DATA_TYPE) * NX);
	cudaMalloc((void **)&s_gpu, sizeof(DATA_TYPE) * NY);
	cudaMalloc((void **)&p_gpu, sizeof(DATA_TYPE) * NY);
	cudaMalloc((void **)&q_gpu, sizeof(DATA_TYPE) * NX);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
	}

	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(r_gpu, r, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice);
	cudaMemcpy(s_gpu, s, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(p_gpu, p, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(q_gpu, q, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice);

	

	t_start = rtclock();
	bicg_kernel1<<< grid1, block >>>(A_gpu, r_gpu, s_gpu);
	cudaDeviceSynchronize();
	cudaFree(r_gpu);
	pre_bemps_free(tid, sizeof(DATA_TYPE) * NX);

	// cudaEventRecord(*pre);
	// cudaEventSynchronize(*pre);

	bicg_kernel2<<< grid2, block >>>(A_gpu, p_gpu, q_gpu);
	cudaDeviceSynchronize();
	cudaFree(A_gpu);
	cudaFree(p_gpu);
	pre_bemps_free(tid, sizeof(DATA_TYPE) * (NX * NY + NY));



	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	
	cudaMemcpy(s_outputFromGpu, s_gpu, sizeof(DATA_TYPE) * NY, cudaMemcpyDeviceToHost);
	cudaMemcpy(q_outputFromGpu, q_gpu, sizeof(DATA_TYPE) * NX, cudaMemcpyDeviceToHost);

	// cudaFree(A_gpu);
	// cudaFree(r_gpu);
	cudaFree(s_gpu);
	// cudaFree(p_gpu);
	cudaFree(q_gpu);
	bemps_free(tid);

	// cudaEventRecord(*stop);
	// cudaEventSynchronize(*stop);
}


int main(int argc, char** argv)
{
	auto start_time = std::chrono::high_resolution_clock::now();
	int tid = atoi(argv[1]);
	std::cout << "tid: " << tid << ", chrono (start): " << start_time.time_since_epoch().count() << std::endl;
	double t_start, t_end;

	// cudaEvent_t start, pre, stop;
	// cudaEventCreate(&start);
	// cudaEventCreate(&pre);
	// cudaEventCreate(&stop);

	// cudaEventRecord(start);

	DATA_TYPE* A;
	DATA_TYPE* r;
	DATA_TYPE* s;
	DATA_TYPE* p;
	DATA_TYPE* q;
	DATA_TYPE* s_outputFromGpu;
	DATA_TYPE* q_outputFromGpu;
 	
	A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	r = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));
	s = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	p = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	q = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));
	s_outputFromGpu = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	q_outputFromGpu = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));

	init_array(A, p, r);

	GPU_argv_init();

	bicgCuda(tid, A, r, s, p, q, s_outputFromGpu, q_outputFromGpu);

	// float totalTime, preTime;
	// cudaEventElapsedTime(&totalTime, start, stop);
	// cudaEventElapsedTime(&preTime, start, pre);

	// printf("total time: %f, pre time: %f\n", totalTime, preTime);

	t_start = rtclock();
	// bicg_cpu(A, r, s, p, q);
	t_end = rtclock();

	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(s, s_outputFromGpu, q, q_outputFromGpu);

	free(A);
	free(r);
	free(s);
	free(p);
	free(q);
	free(s_outputFromGpu);
	free(q_outputFromGpu);

	auto end_time = std::chrono::high_resolution_clock::now();
	std::cout << "tid: " << tid << ", chrono (end): " << start_time.time_since_epoch().count() << std::endl;
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
	std::cout << "tid: " << tid << ", chrono (start to end): " << duration.count() << std::endl;

  	return 0;
}

