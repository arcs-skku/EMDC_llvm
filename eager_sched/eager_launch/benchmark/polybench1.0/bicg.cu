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
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <bitset>
#include <bemps.hpp>
#include <chrono>

using namespace std::chrono;
 
 
#include "../../common/polybenchUtilFuncts.h"
 
//Error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5
 
#define GPU_DEVICE 0
 
# define page_size 4096
# define VA_block 2097152

# define kernel_num 2

/* Problem size. */
#define NX 32768
#define NY 32768
 
/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 32
 
#ifndef M_PI
#define M_PI 3.14159
#endif
 
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
	size_t alloc_size; // 디바이스 메모리에 올릴 페이지 크기
	std::bitset<kernel_num> bit; // liveness check
};

std::vector<Parameters> mem_list;
 
void init_array(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *r, DATA_TYPE *q, DATA_TYPE *s)
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
			q[i] = 0;
			s[i] = 0;
	}
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
		int j;
		for(j=0; j < NY; j++)
		{
			q[i] += A[i * NY + j] * p[j];
		}
	}
}

void task_monitoring(cudaEvent_t event, int tid, long orig_alloc_mem, size_t membytes){
	long update_mem = 0;
	long tmp_mem = 0;
	update_mem = bemps_extra_task_mem(tid);
	tmp_mem = update_mem;
	if(full != 1){
		while(1){
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
				// for(Parameters ret : mem_list){
				// 	ret.alloc_size = update_mem / 5;
				// 	CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
				// 	CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, 0, s_e));
				// }
				// if(full == 1)
				// 	break;
				if(full == 1){
					printf("Hello\n");
					CUDA_CHECK(cudaMemAdvise(mem_list[0].devPtr, sizeof(DATA_TYPE) * NX * NY, mem_list[0].advice, mem_list[0].device));
					CUDA_CHECK(cudaMemPrefetchAsync(mem_list[0].devPtr, sizeof(DATA_TYPE) * NX * NY, 0, s_e));
					CUDA_CHECK(cudaMemAdvise(mem_list[1].devPtr, sizeof(DATA_TYPE) * NX, mem_list[1].advice, mem_list[1].device));
					CUDA_CHECK(cudaMemPrefetchAsync(mem_list[1].devPtr, sizeof(DATA_TYPE) * NX, 0, s_e));
					CUDA_CHECK(cudaMemAdvise(mem_list[2].devPtr, sizeof(DATA_TYPE) * NY, mem_list[2].advice, mem_list[2].device));
					CUDA_CHECK(cudaMemPrefetchAsync(mem_list[2].devPtr, sizeof(DATA_TYPE) * NY, 0, s_e));
					CUDA_CHECK(cudaMemAdvise(mem_list[3].devPtr, sizeof(DATA_TYPE) * NY, mem_list[3].advice, mem_list[3].device));
					CUDA_CHECK(cudaMemPrefetchAsync(mem_list[3].devPtr, sizeof(DATA_TYPE) * NY, 0, s_e));
					CUDA_CHECK(cudaMemAdvise(mem_list[4].devPtr, sizeof(DATA_TYPE) * NX, mem_list[4].advice, mem_list[4].device));
					CUDA_CHECK(cudaMemPrefetchAsync(mem_list[4].devPtr, sizeof(DATA_TYPE) * NX, 0, s_e));
					break;
				}
			}
			CUDA_CHECK(cudaStreamDestroy(s_e));
		}
	}
}

int main(int argc, char** argv)
{
	cudaFree(0);
	printf("Start in GPU Pinned\n");
 
	DATA_TYPE* A;
	DATA_TYPE* r;
	DATA_TYPE* s;
	DATA_TYPE* p;
	DATA_TYPE* q;
	
	Parameters ret1;
	Parameters ret2;
	Parameters ret3;
	Parameters ret4;
	Parameters ret5;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);

	int tid = atoi(argv[1]);
	printf("tid: %d\n", tid);
    size_t membytes = 0;
	membytes += sizeof(DATA_TYPE) * NX * NY;
	membytes += sizeof(DATA_TYPE) * NX;
	membytes += sizeof(DATA_TYPE) * NY;
	membytes += sizeof(DATA_TYPE) * NY;
	membytes += sizeof(DATA_TYPE) * NX;

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

	long alloc_mem = NX*NY*sizeof(DATA_TYPE);
	long other_alloc_mem = NY*sizeof(DATA_TYPE);

	if(full == 0){
		other_alloc_mem = VA_block;
		if(orig_alloc_mem >= NX*NY*sizeof(DATA_TYPE))
			alloc_mem = NX*NY*sizeof(DATA_TYPE);
		else
			alloc_mem = orig_alloc_mem;
	}

	CUDA_CHECK(cudaMallocManaged(&A, NX*NY*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&r, NX*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&s, NY*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&p, NY*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&q, NX*sizeof(DATA_TYPE)));
	
 	init_array(A, p, r, q, s);
 	GPU_argv_init();
 
	ret1.devPtr = A;
	ret1.count = alloc_mem / VA_block;
	ret1.advice = cudaMemAdviseSetPreferredLocation;
	ret1.device = 0;
	ret1.alloc_size = alloc_mem;
 
	ret2.devPtr = r;
	ret2.count = alloc_mem / VA_block;
	ret2.advice = cudaMemAdviseSetPreferredLocation;
	ret2.device = 0;
	ret2.alloc_size = other_alloc_mem;
	 
	ret3.devPtr = s;
	ret3.count = alloc_mem / VA_block;
	ret3.advice = cudaMemAdviseSetPreferredLocation;
	ret3.device = 0;
	ret3.alloc_size = other_alloc_mem;
 
	ret4.devPtr = p;
	ret4.count = alloc_mem / VA_block;
	ret4.advice = cudaMemAdviseSetPreferredLocation;
	ret4.device = 0;
	ret4.alloc_size = other_alloc_mem;

	ret5.devPtr = q;
	ret5.count = alloc_mem / VA_block;
	ret5.advice = cudaMemAdviseSetPreferredLocation;
	ret5.device = 0;
	ret5.alloc_size = other_alloc_mem;

	mem_list.push_back(ret1);
	mem_list.push_back(ret2);
	mem_list.push_back(ret3);
	mem_list.push_back(ret4);
	mem_list.push_back(ret5);

	if(full == 1){
		for(Parameters ret : mem_list){
			CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
			CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, 0, 0));
		}
	}
	// else{
	// 	for(Parameters ret : mem_list){
	// 		CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, 0, 0));
	// 	}
	// }

	cudaStream_t s1;
	CUDA_CHECK(cudaStreamCreate(&s1));

	cudaEvent_t event1;
	CUDA_CHECK(cudaEventCreateWithFlags(&event1, cudaEventDisableTiming));

	bicg_kernel1<<< grid1, block, 0, s1 >>>(A, r, s);

	CUDA_CHECK(cudaEventRecord(event1, s1));

	task_monitoring(event1, tid, orig_alloc_mem, membytes);

	CUDA_CHECK(cudaDeviceSynchronize());

	// cudaFree(r);
	// cudaFree(s);

	// if(full == 1){
	// 	pre_bemps_free(tid, sizeof(DATA_TYPE) * NX * 1);
	// }
	cudaStream_t s2;
	CUDA_CHECK(cudaStreamCreate(&s2));

	cudaEvent_t event2;
	CUDA_CHECK(cudaEventCreateWithFlags(&event2, cudaEventDisableTiming));

	bicg_kernel2<<< grid2, block, 0, s2 >>>(A, p, q);

	CUDA_CHECK(cudaEventRecord(event2, s2));

	task_monitoring(event2, tid, orig_alloc_mem, membytes);

	CUDA_CHECK(cudaDeviceSynchronize());
	
	CUDA_CHECK(cudaMemPrefetchAsync(s, sizeof(DATA_TYPE) * NY, cudaCpuDeviceId, 0));
	CUDA_CHECK(cudaMemPrefetchAsync(q, sizeof(DATA_TYPE) * NX, cudaCpuDeviceId, 0));
	CUDA_CHECK(cudaDeviceSynchronize());
 
	printf("Check Val: %f, %f\n", s[0], q[0]);

	cudaFree(A);
	cudaFree(r);
	cudaFree(s);
	cudaFree(p);
	cudaFree(q);
	
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
 
 
