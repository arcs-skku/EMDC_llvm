/**
 * 3mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#define GPU_DEVICE 0

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
# define NI 8192
# define NJ 8192
# define NK 8192
# define NL 8192
# define NM 8192

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
			A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
		}
	}
  
	for (i = 0; i < NK; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			B[i*NJ + j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}
  
	for (i = 0; i < NJ; i++)
	{
		for (j = 0; j < NM; j++)
		{
			C[i*NM + j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}
  
	for (i = 0; i < NM; i++)
	{
		for (j = 0; j < NL; j++)
		{
			D[i*NL + j] = ((DATA_TYPE) i*(j+2)) / NK;
		}
	}
}


void compareResults(DATA_TYPE *G, DATA_TYPE *G_outputFromGpu)
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < NI; i++)
	{
		for (j=0; j < NL; j++)
		{
			if (percentDiff(G[i*NL + j], G_outputFromGpu[i*NL + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
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

	
__global__ void mm3_kernel1(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *E)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NJ))
	{
		int k;
		for(k=0; k < NK; k++)
		{
			E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
			// printf("E[%d]: %f\n", i * NJ + j, E[i * NJ + j]);
		}
	}
}

	
__global__ void mm3_kernel2(DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *F)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NJ) && (j < NL))
	{
		int k;
		for(k=0; k < NM; k++)
		{
			F[i * NL + j] += C[i * NM + k] * D[k * NL +j];
		}
	}
}

	
__global__ void mm3_kernel3(DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NL))
	{
		int k;
		for(k=0; k < NJ; k++)
		{
			G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
		}
	}
}


void mm3_cpu(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
{
	int i,j,k;
	
	/* E := A*B */
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			E[i*NJ + j] = 0;
			for (k = 0; k < NK; ++k)
			{
				E[i*NJ + j] += A[i*NK + k] * B[k*NJ + j];
			}
		}
	}
		
	/* F := C*D */
	for (i = 0; i < NJ; i++)
	{
		for (j = 0; j < NL; j++)
		{
			F[i*NL + j] = 0;
			for (k = 0; k < NM; ++k)
			{
				F[i*NL + j] += C[i*NM + k] * D[k*NL + j];
			}
		}
	}

  	/* G := E*F */
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{
			G[i*NL + j] = 0;
			for (k = 0; k < NJ; ++k)
			{
				G[i*NL + j] += E[i*NJ + k] * F[k*NL + j];
			}
		}
	}
}


void mm3Cuda(int tid, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, DATA_TYPE* F, 
		DATA_TYPE* G, DATA_TYPE* G_outputFromGpu)
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	DATA_TYPE *D_gpu;
	DATA_TYPE *E_gpu;
	DATA_TYPE *F_gpu;
	DATA_TYPE *G_gpu;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NJ) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NI/ ((float)DIM_THREAD_BLOCK_Y) )));
	dim3 grid2((size_t)(ceil( ((float)NL) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NJ/ ((float)DIM_THREAD_BLOCK_Y) )));
	dim3 grid3((size_t)(ceil( ((float)NL) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NI/ ((float)DIM_THREAD_BLOCK_Y) )));

	// int64_t membytes = sizeof(DATA_TYPE) * (int64_t)(NI * NK + NK * NJ + NJ * NM + NM * NL + NI * NJ + NJ * NL + NI * NL);
	int64_t membytes = 0;
	membytes += sizeof(DATA_TYPE) * NI * NK;
	membytes += sizeof(DATA_TYPE) * NI * NK;
	membytes += sizeof(DATA_TYPE) * NI * NK;
	membytes += sizeof(DATA_TYPE) * NI * NK;
	membytes += sizeof(DATA_TYPE) * NI * NK;
	membytes += sizeof(DATA_TYPE) * NI * NK;
	membytes += sizeof(DATA_TYPE) * NI * NK;
	membytes += 305 * 1024  * 1024;
	membytes += 20 * 1024 * 1024;

	// printf("membytes: %ld\n", membytes);
	// printf("%d: LOG_before\n", tid);
	auto bemps_time = std::chrono::high_resolution_clock::now();
	std::cout << "tid: " << tid << ", chrono (bemps): " << bemps_time.time_since_epoch().count() << std::endl;
	
	bemps_begin(tid, grid1.x, grid1.y, grid1.z, block.x , block.y, block.z, membytes);
	// printf("%d: LOG_after\n", tid);

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NJ * NM);
	cudaMalloc((void **)&D_gpu, sizeof(DATA_TYPE) * NM * NL);
	cudaMalloc((void **)&E_gpu, sizeof(DATA_TYPE) * NI * NJ);
	cudaMalloc((void **)&F_gpu, sizeof(DATA_TYPE) * NJ * NL);
	cudaMalloc((void **)&G_gpu, sizeof(DATA_TYPE) * NI * NL);

	// cudaError_t err = cudaGetLastError();
	// printf("CUDA ERROR: %d, %s\n", err, cudaGetErrorString(err));

	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NJ * NM, cudaMemcpyHostToDevice);
	cudaMemcpy(D_gpu, D, sizeof(DATA_TYPE) * NM * NL, cudaMemcpyHostToDevice);
	cudaMemcpy(E_gpu, E, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(F_gpu, F, sizeof(DATA_TYPE) * NJ * NL, cudaMemcpyHostToDevice);
	cudaMemcpy(G_gpu, G, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyHostToDevice);

	// int flag=0;
	// while(1) {
	// 	if (flag) break;
	// 	scanf("%d", &flag);
	// }
	
	

	t_start = rtclock();
	mm3_kernel1<<<grid1,block>>>(A_gpu, B_gpu, E_gpu);
	cudaDeviceSynchronize();
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	
	// printf("%d: LOG_before_pre1\n", tid);
	pre_bemps_free(tid, (int64_t)sizeof(DATA_TYPE) * ((int64_t)NI * (int64_t)NK +(int64_t) NK *(int64_t) NJ));
	// printf("%d: LOG_after_pre1\n", tid);

	mm3_kernel2<<<grid2,block>>>(C_gpu, D_gpu, F_gpu);
	cudaDeviceSynchronize();
	cudaFree(C_gpu);
	cudaFree(D_gpu);
	// pre_bemps_free(tid, sizeof(DATA_TYPE) * (NJ * NM + NM * NL));
	// printf("%d: LOG_before_pre2\n", tid);
	pre_bemps_free(tid, (int64_t)sizeof(DATA_TYPE) * ((int64_t)NI * (int64_t)NK +(int64_t) NK *(int64_t) NJ));
	// printf("%d: LOG_after_pre2\n", tid);

	// cudaEventRecord(*pre);
	// cudaEventSynchronize(*pre);

	mm3_kernel3<<<grid3,block>>>(E_gpu, F_gpu, G_gpu);
	cudaDeviceSynchronize();
	// cudaFree(E_gpu);
	// cudaFree(F_gpu);
	// pre_bemps_free(tid, sizeof(DATA_TYPE) * (NI * NJ + NJ * NL));
	// printf("%d: LOG_before_pre3\n", tid);
	// pre_bemps_free(tid, (int64_t)sizeof(DATA_TYPE) * ((int64_t)NI * (int64_t)NK +(int64_t) NK *(int64_t) NJ));
	// printf("%d: LOG_after_pre3\n", tid);

	t_end = rtclock();
	cudaMemcpy(G_outputFromGpu, G_gpu, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyDeviceToHost);

	// fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	
	// cudaFree(A_gpu);
	// cudaFree(B_gpu);
	// cudaFree(C_gpu);
	// cudaFree(D_gpu);
	cudaFree(E_gpu);
	cudaFree(F_gpu);
	cudaFree(G_gpu);

	// printf("%d: LOG_before_end\n", tid);
	bemps_free(tid);

	// cudaEventRecord(*stop);
	// cudaEventSynchronize(*stop);
	// printf("%d: LOG_after_end\n", tid);
}


int main(int argc, char** argv)
{
	// cudaEvent_t start, stop;
	// cudaEventCreate(&start);
	// cudaEventCreate(&stop);

	// cudaEventRecord(start);

	auto start_time = std::chrono::high_resolution_clock::now();
	int tid = atoi(argv[1]);
	std::cout << "tid: " << tid << ", chrono (start): " << start_time.time_since_epoch().count() << std::endl;

	// dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	// dim3 grid1((size_t)(ceil( ((float)NJ) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NI/ ((float)DIM_THREAD_BLOCK_Y) )));
	// dim3 grid2((size_t)(ceil( ((float)NL) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NJ/ ((float)DIM_THREAD_BLOCK_Y) )));
	// dim3 grid3((size_t)(ceil( ((float)NL) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NI/ ((float)DIM_THREAD_BLOCK_Y) )));

	// int64_t membytes = 0;
	// membytes += sizeof(DATA_TYPE) * NI * NK;
	// membytes += sizeof(DATA_TYPE) * NI * NK;
	// membytes += sizeof(DATA_TYPE) * NI * NK;
	// membytes += sizeof(DATA_TYPE) * NI * NK;
	// membytes += sizeof(DATA_TYPE) * NI * NK;
	// membytes += sizeof(DATA_TYPE) * NI * NK;
	// membytes += sizeof(DATA_TYPE) * NI * NK;
	// membytes += 305 * 1024  * 1024;
	// membytes += 20 * 1024 * 1024;

	// auto cur_time = std::chrono::high_resolution_clock::now();
	// auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(cur_time.time_since_epoch()).count();
	// std::cout << "TIME STAMP (BEMPS_BEGIN). time: " << nanoseconds << " ns, "  << "tid: " << id << "\n";

	// bemps_begin(tid, grid1.x, grid1.y, grid1.z, block.x , block.y, block.z, membytes);

	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* C;
	DATA_TYPE* D;
	DATA_TYPE* E;
	DATA_TYPE* F;
	DATA_TYPE* G;
	DATA_TYPE* G_outputFromGpu;

	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));
	C = (DATA_TYPE*)malloc(NJ*NM*sizeof(DATA_TYPE));
	D = (DATA_TYPE*)malloc(NM*NL*sizeof(DATA_TYPE));
	E = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	F = (DATA_TYPE*)malloc(NJ*NL*sizeof(DATA_TYPE));
	G = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));
	G_outputFromGpu = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));

	init_array(A, B, C, D);

	// cudaEventRecord(stop);
	// cudaEventSynchronize(stop);

	// float time;

	// cudaEventElapsedTime(&time, start, stop);

	// printf("ELAPSED TIME: %f\n", time);

	// cudaEventDestroy(start);
	// cudaEventDestroy(stop);
	

	// GPU_argv_init();

	mm3Cuda(tid, A, B, C, D, E, F, G, G_outputFromGpu);

	// float totalTime, preTime;
	// cudaEventElapsedTime(&totalTime, start, stop);
	// cudaEventElapsedTime(&preTime, start, pre);


	// printf("total time: %f, pre time: %f\n", totalTime, preTime);

	t_start = rtclock();

	// mm3_cpu(A, B, C, D, E, F, G);
	
	t_end = rtclock();

	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(G, G_outputFromGpu);

	free(A);
	free(B);
	free(C);
	free(D);
	free(E);
	free(F);
	free(G);
	free(G_outputFromGpu);

	auto end_time = std::chrono::high_resolution_clock::now();
	std::cout << "tid: " << tid << ", chrono (end): " << start_time.time_since_epoch().count() << std::endl;
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
	std::cout << "tid: " << tid << ", chrono (start to end): " << duration.count() << std::endl;

	// bemps_free(tid);

	// cudaEventDestroy(start);
    // cudaEventDestroy(pre);
    // cudaEventDestroy(stop);

	// printf("%d: LOG_before_return\n", tid);
	return 0;
}
