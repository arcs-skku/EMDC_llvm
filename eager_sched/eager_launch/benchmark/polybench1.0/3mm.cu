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
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <bitset>
#include <bemps.hpp>

using namespace std::chrono;

#include "../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
# define NI 16384
# define NJ 16384
# define NK 16384
# define NL 16384
# define NM 16384

# define page_size 4096
# define VA_block 2097152

# define kernel_num 3

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 32

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

bool extra_status; // read from channel
size_t extra_mem; // read from channel

bool full; // 1 = fully secured

#define CUDA_CHECK(val) { \
	if (val != cudaSuccess) { \
		fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(val), __LINE__, __FILE__); \
		exit(val); \
	} \
}

struct Parameters{
	void* devPtr;
	size_t count;
	cudaMemoryAdvise advice;
	int device;
	long alloc_size; // 디바이스 메모리에 올릴 페이지 크기
	std::bitset<kernel_num> bit; // liveness check
};

std::vector<Parameters> mem_list;

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, DATA_TYPE* F, DATA_TYPE* G)
{
	int i, j;

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NK; j++)
		{
			A[i*NK + j] = 1;
			B[i*NJ + j] = 1;
			C[i*NM + j] = 1;
			D[i*NL + j] = 1;
			E[i*NJ + j] = 0;
			F[i*NL + j] = 0;
			G[i*NL + j] = 0;
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

void task_monitoring(cudaEvent_t event, int tid, long orig_alloc_mem, size_t membytes){
	while(1){
		long update_mem = 0;
		bool chk_former_task = 0;
		update_mem = bemps_extra_task_mem(tid);
		cudaStream_t s_e;
		CUDA_CHECK(cudaStreamCreate(&s_e));
		if(orig_alloc_mem != update_mem){
			chk_former_task = 1;
		}
		if(cudaEventQuery(event) == cudaSuccess){
			printf("Kernel End\n");
			break;
		}
		if((chk_former_task == 1) && (full != 1)){
			if(update_mem == membytes){
				full = 1;
			}
			for(Parameters ret : mem_list){
				ret.alloc_size = update_mem / 7;
				if(ret.alloc_size % 2097152 != 0){
                    ret.alloc_size = long(ret.alloc_size / 2097152) * 2097152;
                }
				printf("%ld\n", ret.alloc_size);
				CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
				CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, 0, s_e));
			}
			if(full == 1)
				break;
		}
		CUDA_CHECK(cudaStreamDestroy(s_e));
	}
}

int main(int argc, char** argv)
{
	cudaFree(0);
	printf("Start in GPU Pinned\n");
	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* C;
	DATA_TYPE* D;
	DATA_TYPE* E;
	DATA_TYPE* F;
	DATA_TYPE* G;

	Parameters ret1;
	Parameters ret2;
	Parameters ret3;
	Parameters ret4;
	Parameters ret5;
	Parameters ret6;
	Parameters ret7;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NJ) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NI/ ((float)DIM_THREAD_BLOCK_Y) )));
	dim3 grid2((size_t)(ceil( ((float)NL) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NJ/ ((float)DIM_THREAD_BLOCK_Y) )));
	dim3 grid3((size_t)(ceil( ((float)NL) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NI/ ((float)DIM_THREAD_BLOCK_Y) )));

	int tid = atoi(argv[1]);
    size_t membytes = 0;
	membytes += sizeof(DATA_TYPE) * NI * NK;
	membytes += sizeof(DATA_TYPE) * NK * NJ;
	membytes += sizeof(DATA_TYPE) * NJ * NM;
	membytes += sizeof(DATA_TYPE) * NM * NL;
	membytes += sizeof(DATA_TYPE) * NI * NJ;
	membytes += sizeof(DATA_TYPE) * NJ * NL;
	membytes += sizeof(DATA_TYPE) * NI * NL;

	struct timespec specific_time;
  struct tm *now;
  int millsec;
  clock_gettime( CLOCK_REALTIME, &specific_time);
  now = localtime(&specific_time.tv_sec);
  millsec = specific_time.tv_nsec;

  millsec = floor (specific_time.tv_nsec/1.0e6);


  printf("TID: %d before schedule, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
        now->tm_mon + 1, now->tm_mday, now->tm_hour, 
        now->tm_min, now->tm_sec, millsec);

	auto t_start = high_resolution_clock::now();

  long orig_alloc_mem = bemps_begin(tid, grid1.x, grid1.y, grid1.z, block.x, block.y, block.z, membytes);
	
	clock_gettime( CLOCK_REALTIME, &specific_time);
    now = localtime(&specific_time.tv_sec);
    millsec = specific_time.tv_nsec;

    millsec = floor (specific_time.tv_nsec/1.0e6);

    printf("TID: %d after schedule, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
        now->tm_mon + 1, now->tm_mday, now->tm_hour, 
        now->tm_min, now->tm_sec, millsec);

	if (membytes <= orig_alloc_mem)
		full = 1;
	else
		full = 0;
	printf("Full: %d\n", full);
	
	long alloc_mem = orig_alloc_mem / 7;

	printf("alloc_mem: %ld\n", alloc_mem);
	CUDA_CHECK(cudaMallocManaged(&A, NI*NK*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&B, NK*NJ*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&C, NI*NJ*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&D, NM*NL*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&E, NI*NJ*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&F, NJ*NL*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&G, NI*NL*sizeof(DATA_TYPE)));

	init_array(A, B, C, D, E, F, G);
	GPU_argv_init();

	ret1.devPtr = A;
	ret1.count = alloc_mem / VA_block;
	ret1.advice = cudaMemAdviseSetPreferredLocation;
	ret1.device = 0;
	ret1.alloc_size = alloc_mem;

	ret2.devPtr = B;
	ret2.count = alloc_mem / VA_block;
	ret2.advice = cudaMemAdviseSetPreferredLocation;
	ret2.device = 0;
	ret2.alloc_size = alloc_mem;

	ret3.devPtr = C;
	ret3.count = alloc_mem / VA_block;
	ret3.advice = cudaMemAdviseSetPreferredLocation;
	ret3.device = 0;
	ret3.alloc_size = alloc_mem;

	ret4.devPtr = D;
	ret4.count = alloc_mem / VA_block;
	ret4.advice = cudaMemAdviseSetPreferredLocation;
	ret4.device = 0;
	ret4.alloc_size = alloc_mem;

	ret5.devPtr = E;
	ret5.count = alloc_mem / VA_block;
	ret5.advice = cudaMemAdviseSetPreferredLocation;
	ret5.device = 0;
	ret5.alloc_size = alloc_mem;

	ret6.devPtr = F;
	ret6.count = alloc_mem / VA_block;
	ret6.advice = cudaMemAdviseSetPreferredLocation;
	ret6.device = 0;
	ret6.alloc_size = alloc_mem;

	ret7.devPtr = G;
	ret7.count = alloc_mem / VA_block;
	ret7.advice = cudaMemAdviseSetPreferredLocation;
	ret7.device = 0;
	ret7.alloc_size = alloc_mem;

	mem_list.push_back(ret1);
	mem_list.push_back(ret2);
	mem_list.push_back(ret3);
	mem_list.push_back(ret4);
	mem_list.push_back(ret5);
	mem_list.push_back(ret6);
	mem_list.push_back(ret7);

	for(Parameters ret : mem_list){
		CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
		CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, 0, 0));
	}

	cudaStream_t s1;
	CUDA_CHECK(cudaStreamCreate(&s1));

	cudaEvent_t event1;
	CUDA_CHECK(cudaEventCreateWithFlags(&event1, cudaEventDisableTiming));

	mm3_kernel1<<<grid1,block, 0, s1>>>(A, B, E);
	
	CUDA_CHECK(cudaEventRecord(event1, s1));

	task_monitoring(event1, tid, orig_alloc_mem, membytes);

	CUDA_CHECK(cudaDeviceSynchronize());

	cudaStream_t s2;
	CUDA_CHECK(cudaStreamCreate(&s2));

	cudaEvent_t event2;
	CUDA_CHECK(cudaEventCreateWithFlags(&event2, cudaEventDisableTiming));

	mm3_kernel2<<<grid2,block, 0, s2>>>(C, D, F);

	CUDA_CHECK(cudaEventRecord(event2, s2));

	task_monitoring(event2, tid, orig_alloc_mem, membytes);

	CUDA_CHECK(cudaDeviceSynchronize());

	cudaStream_t s3;
	CUDA_CHECK(cudaStreamCreate(&s3));

	cudaEvent_t event3;
	CUDA_CHECK(cudaEventCreateWithFlags(&event3, cudaEventDisableTiming));

	mm3_kernel3<<<grid3,block, 0, s3>>>(E, F, G);
	
	CUDA_CHECK(cudaEventRecord(event3, s3));

	task_monitoring(event3, tid, orig_alloc_mem, membytes);

	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemPrefetchAsync(G, sizeof(DATA_TYPE) * NI * NL, cudaCpuDeviceId, 0));
	CUDA_CHECK(cudaDeviceSynchronize());

	printf("Check Val: %f\n", G[0]);
 
	CUDA_CHECK(cudaFree(A));
	CUDA_CHECK(cudaFree(B));
	CUDA_CHECK(cudaFree(C));
	CUDA_CHECK(cudaFree(D));
	CUDA_CHECK(cudaFree(E));
	CUDA_CHECK(cudaFree(F));
	CUDA_CHECK(cudaFree(G));
	 
	bemps_free(tid);

	clock_gettime( CLOCK_REALTIME, &specific_time);
  now = localtime(&specific_time.tv_sec);
  millsec = specific_time.tv_nsec;

  millsec = floor (specific_time.tv_nsec/1.0e6);


  printf("TID: %d finish work, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
        now->tm_mon + 1, now->tm_mday, now->tm_hour, 
        now->tm_min, now->tm_sec, millsec);

	auto t_stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(t_stop - t_start);
	std::cout << "Total Time: " << duration.count() << std::endl;

	return 0;
}

