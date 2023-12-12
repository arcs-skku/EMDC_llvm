/**
 * mvt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#define N 32768
// #define N 65536

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


void init_array(DATA_TYPE* A, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2)
{
	int i, j;

	for (i = 0; i < N; i++)
	{
		x1[i] = ((DATA_TYPE) i) / N;
		x2[i] = ((DATA_TYPE) i + 1) / N;
		y1[i] = ((DATA_TYPE) i + 3) / N;
		y2[i] = ((DATA_TYPE) i + 4) / N;
		for (j = 0; j < N; j++)
		{
			A[i*N + j] = ((DATA_TYPE) i*j) / N;
		}
	}
}



void runMvt(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2)
{
	int i, j;
	
	for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++) 
		{
       			x1[i] = x1[i] + a[i*N + j] * y1[j];
        	}
    	}
	
	for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++) 
		{
 		       	x2[i] = x2[i] + a[j*N + i] * y2[j];
      		}
    	}
}


void compareResults(DATA_TYPE* x1, DATA_TYPE* x1_outputFromGpu, DATA_TYPE* x2, DATA_TYPE* x2_outputFromGpu)
{
	int i, fail;
	fail = 0;
	
	for (i=0; i<N; i++) 
	{
		if (percentDiff(x1[i], x1_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}

		if (percentDiff(x2[i], x2_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void mvt_kernel1(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *y_1)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		int j;
		for(j=0; j < N; j++)
		{
			x1[i] += a[i * N + j] * y_1[j];
		}
	}
}


__global__ void mvt_kernel2(DATA_TYPE *a, DATA_TYPE *x2, DATA_TYPE *y_2)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		int j;
		for(j=0; j < N; j++)
		{
			x2[i] += a[j * N + i] * y_2[j];	
		}
	}
}

void mvtCuda(int tid, DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y_1, DATA_TYPE* y_2, 
			DATA_TYPE* x1_outputFromGpu, DATA_TYPE* x2_outputFromGpu)
{
	double t_start, t_end;

	DATA_TYPE* a_gpu;
	DATA_TYPE* x1_gpu;
	DATA_TYPE* x2_gpu;
	DATA_TYPE* y_1_gpu;
	DATA_TYPE* y_2_gpu;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)ceil((float)N/ ((float)DIM_THREAD_BLOCK_X)), 1);

	int64_t membytes = 0;
	membytes += sizeof(DATA_TYPE) * N * N;
	membytes += sizeof(DATA_TYPE) * N;
	membytes += sizeof(DATA_TYPE) * N;
	membytes += sizeof(DATA_TYPE) * N;
	membytes += sizeof(DATA_TYPE) * N;
	membytes += 305 * 1024 * 1024;
	membytes += 20 * 1024 * 1024;

	auto bemps_time = std::chrono::high_resolution_clock::now();
	std::cout << "tid: " << tid << ", chrono (bemps): " << bemps_time.time_since_epoch().count() << std::endl;

	bemps_begin(tid, grid.x, grid.y, grid.z, block.x, block.y, block.z, membytes);

	cudaMalloc((void **)&a_gpu, sizeof(DATA_TYPE) * N * N);
	cudaMalloc((void **)&x1_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&x2_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&y_1_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&y_2_gpu, sizeof(DATA_TYPE) * N);
	cudaMemcpy(a_gpu, a, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(x1_gpu, x1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(x2_gpu, x2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_1_gpu, y_1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_2_gpu, y_2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	
	// int flag=0;
	// while(1) {
	// 	if (flag)	break;
	// 	scanf("%d", &flag);
	// }
	
	t_start = rtclock();
	mvt_kernel1<<<grid,block>>>(a_gpu,x1_gpu,y_1_gpu);
	cudaDeviceSynchronize();
	cudaMemcpy(x1_outputFromGpu, x1_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);

	// Eager Free
	cudaFree(x1_gpu);
	cudaFree(y_1_gpu);
	pre_bemps_free(tid, sizeof(DATA_TYPE) * (N + N));

	// cudaEventRecord(*pre);
	// cudaEventSynchronize(*pre);

	mvt_kernel2<<<grid,block>>>(a_gpu,x2_gpu,y_2_gpu);
	cudaDeviceSynchronize();

	cudaFree(a_gpu);
	cudaFree(y_2_gpu);
	pre_bemps_free(tid, sizeof(DATA_TYPE) * (N * N + N));
	cudaMemcpy(x2_outputFromGpu, x2_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);    
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	
	
	// cudaFree(a_gpu);
	// cudaFree(x1_gpu);
	cudaFree(x2_gpu);
	// cudaFree(y_1_gpu);
	cudaFree(y_2_gpu);

	bemps_free(tid);

	// cudaEventRecord(*stop);
	// cudaEventSynchronize(*stop);
}


int main(int argc, char* argv[])
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

	DATA_TYPE* a;
	DATA_TYPE* x1;
	DATA_TYPE* x2;
	DATA_TYPE* x1_outputFromGpu;
	DATA_TYPE* x2_outputFromGpu;
	DATA_TYPE* y_1;
	DATA_TYPE* y_2;

	a = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
	x1 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	x2 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	x1_outputFromGpu = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	x2_outputFromGpu = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	y_1 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	y_2 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));

	init_array(a, x1, x2, y_1, y_2);
	
	GPU_argv_init();

	mvtCuda(tid, a, x1, x2, y_1, y_2, x1_outputFromGpu, x2_outputFromGpu);

	// float totalTime, preTime;
	// cudaEventElapsedTime(&totalTime, start, stop);
	// cudaEventElapsedTime(&preTime, start, pre);

	// printf("total time: %f, pre time: %f\n", totalTime, preTime);
	
	t_start = rtclock();

	//run the algorithm on the CPU
	// runMvt(a, x1, x2, y_1, y_2);

	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(x1, x1_outputFromGpu, x2, x2_outputFromGpu);

	free(a);
	free(x1);
	free(x2);
	free(x1_outputFromGpu);
	free(x2_outputFromGpu);
	free(y_1);
	free(y_2);

	auto end_time = std::chrono::high_resolution_clock::now();
	std::cout << "tid: " << tid << ", chrono (end): " << start_time.time_since_epoch().count() << std::endl;
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
	std::cout << "tid: " << tid << ", chrono (start to end): " << duration.count() << std::endl;

  	return 0;
}

