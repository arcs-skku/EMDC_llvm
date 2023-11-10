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
#include <bemps.hpp>

using namespace std::chrono;

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size. */
# define NI 16384
# define NJ 16384
# define NK 16384
# define NL 16384

# define page_size 4096
# define VA_block 2097152

# define kernel_num 2

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
				ret.alloc_size = update_mem / 5;
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
	printf("Start in GPU Pinned\n");
	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* C;
	DATA_TYPE* D;
	DATA_TYPE* E;

	Parameters ret1;
	Parameters ret2;
	Parameters ret3;
	Parameters ret4;
	Parameters ret5;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)ceil( ((float)NJ) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );

  int tid = atoi(argv[1]);
  size_t membytes = 0;
	membytes += sizeof(DATA_TYPE) * NI * NK;
	membytes += sizeof(DATA_TYPE) * NK * NJ;
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
	
	long alloc_mem = orig_alloc_mem / 5;

	size_t req_mem = 1;
	size_t free_mem = 2;

	CUDA_CHECK(cudaMallocManaged(&A, NI*NK*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&B, NK*NJ*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&C, NI*NJ*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&D, NJ*NL*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&E, NI*NL*sizeof(DATA_TYPE)));

	auto init_start = high_resolution_clock::now();
  	init_array(A, B, C, D, E);
	auto init_stop = high_resolution_clock::now();
	GPU_argv_init();

	ret1.devPtr = A;
	ret1.count = alloc_mem / VA_block;
	ret1.advice = cudaMemAdviseSetPreferredLocation;
	ret1.device = 0;
	ret1.alloc_size = alloc_mem;
	ret1.bit.set(0, 1);
	ret1.bit.set(1, 0);

	ret2.devPtr = B;
	ret2.count = alloc_mem / VA_block;
	ret2.advice = cudaMemAdviseSetPreferredLocation;
	ret2.device = 0;
	ret2.alloc_size = alloc_mem;
	ret2.bit.set(0, 1);
	ret2.bit.set(1, 0);

	ret3.devPtr = C;
	ret3.count = alloc_mem / VA_block;
	ret3.advice = cudaMemAdviseSetPreferredLocation;
	ret3.device = 0;
	ret3.alloc_size = alloc_mem;
	ret3.bit.set(0, 1);
	ret3.bit.set(1, 1);

	ret4.devPtr = D;
	ret4.count = alloc_mem / VA_block;
	ret4.advice = cudaMemAdviseSetPreferredLocation;
	ret4.device = 0;
	ret4.alloc_size = alloc_mem;
	ret4.bit.set(0, 0);
	ret4.bit.set(1, 1);

	ret5.devPtr = E;
	ret5.count = alloc_mem / VA_block;
	ret5.advice = cudaMemAdviseSetPreferredLocation;
	ret5.device = 0;
	ret5.alloc_size = alloc_mem;
	ret5.bit.set(0, 0);
	ret5.bit.set(1, 1);

	mem_list.push_back(ret1);
	mem_list.push_back(ret2);
	mem_list.push_back(ret3);
	mem_list.push_back(ret4);
	mem_list.push_back(ret5);

	for(Parameters ret : mem_list){
		CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
		CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, 0, 0));
	}
	
	cudaStream_t s1;
	cudaStream_t s2;
	cudaStream_t s3;
	
	CUDA_CHECK(cudaStreamCreate(&s1));
	CUDA_CHECK(cudaStreamCreate(&s2));
	CUDA_CHECK(cudaStreamCreate(&s3));
	
	cudaEvent_t event1;
	CUDA_CHECK(cudaEventCreateWithFlags(&event1, cudaEventDisableTiming));

	mm2_kernel1<<<grid1, block, 0, s1>>>(A, B, C);

	CUDA_CHECK(cudaEventRecord(event1, s1));

	task_monitoring(event1, tid, orig_alloc_mem, membytes);
	
	CUDA_CHECK(cudaDeviceSynchronize());
	
	cudaEvent_t event2;
	CUDA_CHECK(cudaEventCreateWithFlags(&event2, cudaEventDisableTiming));

	mm2_kernel1<<<grid1,block, 0, s2>>>(C, D, E);

	CUDA_CHECK(cudaEventRecord(event2, s2));

	task_monitoring(event2, tid, orig_alloc_mem, membytes);

	CUDA_CHECK(cudaDeviceSynchronize());
	
	CUDA_CHECK(cudaMemPrefetchAsync(E, sizeof(DATA_TYPE) * NI * NL, cudaCpuDeviceId, 0));
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
	
	auto init_duration = duration_cast<microseconds>(init_stop - init_start);
	std::cout << "Init Time: " << init_duration.count() << std::endl;
	printf("Stop in GPU Pinned\n");

	// std::cout << ret1.bit[1] << ret1.bit[0] << std::endl;
	// std::cout << ret2.bit[1] << ret2.bit[0] << std::endl;
	// std::cout << ret3.bit[1] << ret3.bit[0] << std::endl;
	// std::cout << ret4.bit[1] << ret4.bit[0] << std::endl;
	// std::cout << ret5.bit[1] << ret5.bit[0] << std::endl;

  	return 0;
}
