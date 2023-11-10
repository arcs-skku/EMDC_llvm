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
#include <bemps.hpp>
using namespace std::chrono;
 
#include "../../common/polybenchUtilFuncts.h"
 
 //define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05
 
#define GPU_DEVICE 0

# define page_size 4096
# define VA_block 2097152

# define kernel_num 1

/* Problem size */
#define NI 32768
#define NJ 32768
 
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
				ret.alloc_size = update_mem / 2;
                if(ret.alloc_size % 2097152 != 0){
                    ret.alloc_size = long(ret.alloc_size / 2097152) * 2097152;
                }
				CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
				CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, 0, s_e));
			}
			if(full == 1)
				break;
		}
		CUDA_CHECK(cudaStreamDestroy(s_e));
	}
}

int main(int argc, char *argv[])
{
    cudaFree(0);
 
    DATA_TYPE* A;
    DATA_TYPE* B;
     
    Parameters ret1;
	Parameters ret2;

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((size_t)ceil( ((float)NI) / ((float)block.x) ), (size_t)ceil( ((float)NJ) / ((float)block.y)) );

    int tid = atoi(argv[1]);
    size_t membytes = 0;
	membytes += sizeof(DATA_TYPE) * NI * NJ;
	membytes += sizeof(DATA_TYPE) * NI * NJ;

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

    long orig_alloc_mem = bemps_begin(tid, grid.x, grid.y, grid.z, block.x, block.y, block.z, membytes);
	
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

    long alloc_mem = orig_alloc_mem / 2;

    CUDA_CHECK(cudaMallocManaged(&A, NI*NJ*sizeof(DATA_TYPE)));
	CUDA_CHECK(cudaMallocManaged(&B, NI*NJ*sizeof(DATA_TYPE)));
 
    //initialize the arrays
    auto init_start = high_resolution_clock::now();
    init(A, B);
    auto init_stop = high_resolution_clock::now();
    GPU_argv_init();

    ret1.devPtr = A;
	ret1.count = alloc_mem / VA_block;
	ret1.advice = cudaMemAdviseSetPreferredLocation;
	ret1.device = 0;
	ret1.alloc_size = alloc_mem;
	// ret1.bit.set(0, 1);
	// ret1.bit.set(1, 0);

	ret2.devPtr = B;
	ret2.count = alloc_mem / VA_block;
	ret2.advice = cudaMemAdviseSetPreferredLocation;
	ret2.device = 0;
	ret2.alloc_size = alloc_mem;
	// ret2.bit.set(0, 1);
	// ret2.bit.set(1, 0);

    mem_list.push_back(ret1);
	mem_list.push_back(ret2);
    
    for(Parameters ret : mem_list){
		CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
		CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, 0, 0));
	}
    
    cudaStream_t s1;
    CUDA_CHECK(cudaStreamCreate(&s1));

    cudaEvent_t event1;
	CUDA_CHECK(cudaEventCreateWithFlags(&event1, cudaEventDisableTiming));

    Convolution2D_kernel<<<grid,block, 0, s1>>>(A, B);

    CUDA_CHECK(cudaEventRecord(event1, s1));

	task_monitoring(event1, tid, orig_alloc_mem, membytes);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemPrefetchAsync(B, sizeof(DATA_TYPE) * NI * NJ, cudaCpuDeviceId, 0));
	CUDA_CHECK(cudaDeviceSynchronize());

    printf("Check val: %f\n", B[0]);
     
    cudaFree(A);
    cudaFree(B);
    
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
     
    return 0;
}
 
 
