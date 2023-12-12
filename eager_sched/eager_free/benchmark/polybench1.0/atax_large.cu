/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <iostream>
#include <bemps.hpp>
#include <chrono>

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size. */
#define NX 32786
#define NY 32786

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

#define ROUND_UP(x, n) (((x + n - 1) / n) * n)

size_t roundup(size_t size, size_t granularity){
    size_t mult = ceil(size/(double)granularity);
    return mult*granularity;
}

static inline void
checkDrvError(CUresult res, const char *tok, const char *file, unsigned line)
{
    if (res != CUDA_SUCCESS) {
        const char *errStr = NULL;
        (void)cuGetErrorString(res, &errStr);
        std::cerr << file << ':' << line << ' ' << tok
                  << "failed (" << (unsigned)res << "): " << errStr << std::endl;
        abort();
    }
}


#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);



void init_array(DATA_TYPE *x, DATA_TYPE *A)
{
	int i, j;

	for (i = 0; i < NX; i++)
	{
		x[i] = i * M_PI;
		for (j = 0; j < NY; j++)
		{
			A[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
		}
	}
}


void compareResults(DATA_TYPE *z, DATA_TYPE *z_outputFromGpu)
{
	int i, fail;
	fail = 0;

	for (i=0; i<NY; i++)
	{
		if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
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


__global__ void atax_kernel1(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NX)
	{
		int j;
		for(j=0; j < NY; j++)
		{
			tmp[i] += A[i * NY + j] * x[j];
		}
	}
}

__global__ void atax_kernel2(DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < NY)
	{
		int i;
		for(i=0; i < NX; i++)
		{
			y[j] += A[i * NY + j] * tmp[i];
		}
	}
}


void atax_cpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp)
{
	int i,j;
	
	for (i= 0; i < NY; i++)
	{
    	y[i] = 0;
	}
  
	for (i = 0; i < NX; i++)
 	{
      	tmp[i] = 0;

      	for (j = 0; j < NY; j++)
		{
			tmp[i] = tmp[i] + A[i*NY + j] * x[j];
		}
		
      	for (j = 0; j < NY; j++)
		{
			y[j] = y[j] + A[i*NY + j] * tmp[i];
		}
    }
}


void ataxGpu(int tid, DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp, DATA_TYPE* y_outputFromGpu)
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);

	int64_t membytes = sizeof(DATA_TYPE) * (NX* NY + NY + NY + NX);
	membytes += 305 * 1024 * 1024;
	membytes += 20 * 1024 * 1024;
	printf("membytes: %ld\n", membytes);

	auto bemps_time = std::chrono::high_resolution_clock::now();
	std::cout << "tid: " << tid << ", chrono (bemps): " << bemps_time.time_since_epoch().count() << std::endl;

	bemps_begin(tid, max(grid1.x, grid2.x), max(grid1.y, grid2.y), max(grid1.z, grid2.z), block.x, block.y, block.z, membytes);
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);


	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NX * NY);
	cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * NY);
	cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * NY);
	cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * NX);

	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice);
	
	t_start = rtclock();
	cudaEventRecord(start);
	atax_kernel1<<< grid1, block >>>(A_gpu,x_gpu,tmp_gpu);
	cudaDeviceSynchronize();

	cudaEventRecord(end);
	cudaEventSynchronize(end);

	float cudaElapsedTime =0.0;
	cudaEventElapsedTime(&cudaElapsedTime, start, end);
	std::cout << "atax_kernel1 elapsed Time: " << cudaElapsedTime << "\n";

	cudaFree(x_gpu);
	pre_bemps_free(tid, sizeof(DATA_TYPE) * NY);

	// cudaEventRecord(*pre);
	// cudaEventSynchronize(*pre);

	atax_kernel2<<< grid2, block >>>(A_gpu,y_gpu,tmp_gpu);
	cudaDeviceSynchronize();
	cudaFree(y_gpu);
	cudaFree(A_gpu);
	pre_bemps_free(tid, sizeof(DATA_TYPE) * (NX * NY + NY));
	

	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	
	cudaMemcpy(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * NX, cudaMemcpyDeviceToHost);

	// cudaFree(A_gpu);
	// cudaFree(x_gpu);
	// cudaFree(y_gpu);
	cudaFree(tmp_gpu);

	bemps_free(tid);

	cudaEventDestroy(start);
	cudaEventDestroy(end);
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
	DATA_TYPE* x;
	DATA_TYPE* y;
	DATA_TYPE* y_outputFromGpu;
	DATA_TYPE* tmp;

	A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	x = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	y = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	y_outputFromGpu = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	tmp = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));

	init_array(x, A);

	GPU_argv_init();
	ataxGpu(tid, A, x, y, tmp, y_outputFromGpu);

	// float totalTime, preTime;
	// cudaEventElapsedTime(&totalTime, start, stop);
	// cudaEventElapsedTime(&preTime, start, pre);

	// printf("total time: %f, pre time: %f\n", totalTime, preTime);

	
	t_start = rtclock();
	// atax_cpu(A, x, y, tmp);
	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(y, y_outputFromGpu);

	free(A);
	free(x);
	free(y);
	free(y_outputFromGpu);
	free(tmp);

	auto end_time = std::chrono::high_resolution_clock::now();
	std::cout << "tid: " << tid << ", chrono (end): " << start_time.time_since_epoch().count() << std::endl;
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
	std::cout << "tid: " << tid << ", chrono (start to end): " << duration.count() << std::endl;

  	return 0;
}

