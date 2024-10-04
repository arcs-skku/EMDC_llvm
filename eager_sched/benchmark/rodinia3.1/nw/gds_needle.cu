#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "needle.h"
#include <cuda.h>
#include <sys/time.h>

#include <bemps.hpp>
#include <algorithm>
#include <vector>
#include <chrono>
#include <iostream>

#include "cufile.h"
#include <unistd.h>
#include <error.h>
#include <fcntl.h>
#include <errno.h>

using namespace std::chrono;

// includes, kernels
#include "needle_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);


int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

bool full; // 1 = fully secured
int priority;

struct Parameters{
	void* devPtr;
	size_t count;
	cudaMemoryAdvise advice;
	int device;
	size_t alloc_size; // 디바이스 메모리에 올릴 페이지 크기
	// std::bitset<kernel_num> bit; // liveness check
};

std::vector<Parameters> mem_list;

#define CUDA_CHECK(val) { \
	if (val != cudaSuccess) { \
		fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(val), __LINE__, __FILE__); \
		exit(val); \
	} \
}

size_t page_align (size_t mem){
	if((mem % 2097152) != 0){
		return (2097152 * (mem / 2097152 + 1));
	}
	else{
		return mem;
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
				if(full == 1){
					priority = -5;
					cudaStream_t s_e;
					CUDA_CHECK(cudaStreamCreateWithPriority(&s_e, 0, priority));
					printf("Hello\n");
					for(Parameters ret : mem_list){
						CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
						CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, ret.device, s_e));
					}
					CUDA_CHECK(cudaStreamSynchronize(s_e));
					CUDA_CHECK(cudaStreamDestroy(s_e));
					break;
				}
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	int tid = atoi(argv[3]);

	struct timespec specific_time;
	struct tm *now;
	int millsec;
	clock_gettime( CLOCK_REALTIME, &specific_time);
	now = localtime(&specific_time.tv_sec);
	millsec = specific_time.tv_nsec;
  
	millsec = floor (specific_time.tv_nsec/1.0e6);
  
	printf("TID: %d Application begin, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
		now->tm_mon + 1, now->tm_mday, now->tm_hour, 
		now->tm_min, now->tm_sec, millsec);

  printf("WG size of kernel = %d \n", BLOCK_SIZE);

    runTest( argc, argv);

	clock_gettime( CLOCK_REALTIME, &specific_time);
	now = localtime(&specific_time.tv_sec);
	millsec = specific_time.tv_nsec;
  
	millsec = floor (specific_time.tv_nsec/1.0e6);
  
	printf("TID: %d Application end, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
		now->tm_mon + 1, now->tm_mday, now->tm_hour, 
		now->tm_min, now->tm_sec, millsec);

    return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> \n", argv[0]);
	fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
	fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
	exit(1);
}

void runTest( int argc, char** argv) 
{
    int max_rows, max_cols, penalty;
    int *input_itemsets, *output_itemsets, *referrence;
	int *matrix_cuda,  *referrence_cuda;
	int size;

	int *tmp_referrence, *tmp_input_itemsets;
	
	int ef_cnt = 0;
  	int ret_dev_id;
	int tid;
	size_t ef_mem;

	Parameters ret1;
	Parameters ret2;
	
    // the lengths of the two sequences should be able to divided by 16.
	// And at current stage  max_rows needs to equal max_cols
	if (argc == 4)
	{
		max_rows = atoi(argv[1]);
		max_cols = atoi(argv[1]);
		penalty = atoi(argv[2]);
		tid = atoi(argv[3]);
	}
    else{
	usage(argc, argv);
    }
	
	if(atoi(argv[1])%16!=0){
	fprintf(stderr,"The dimension values must be a multiple of 16\n");
	exit(1);
	}

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
	size = max_cols * max_rows;

	dim3 dimGrid;
	dim3 dimBlock(BLOCK_SIZE, 1);
	int block_width = ( max_cols - 1 )/BLOCK_SIZE;
	size = max_cols * max_rows;
	printf("block_width: %d\n", block_width);

	// For reducing host initilization time in task

	tmp_referrence = (int *)malloc( max_rows * max_cols * sizeof(int) );
    tmp_input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );

	// if (!input_itemsets)
	// 	fprintf(stderr, "error: can not allocate memory");

    srand ( 7 );
	
	
    // for (int i = 0 ; i < max_cols; i++){
	// 	for (int j = 0 ; j < max_rows; j++){
	// 		tmp_input_itemsets[i*max_cols+j] = 0;
	// 	}
	// }
	
	printf("Start Needleman-Wunsch\n");
	
	// for( int i=1; i< max_rows ; i++){    //please define your own sequence. 
	// 	tmp_input_itemsets[i*max_cols] = rand() % 10 + 1;
	// }
    // for( int j=1; j< max_cols ; j++){    //please define your own sequence.
	// 	tmp_input_itemsets[j] = rand() % 10 + 1;
	// }


	// for (int i = 1 ; i < max_cols; i++){
	// 	for (int j = 1 ; j < max_rows; j++){
	// 		tmp_referrence[i*max_cols+j] = blosum62[tmp_input_itemsets[i*max_cols]][tmp_input_itemsets[j]];
	// 	}
	// }

    // for( int i = 1; i< max_rows ; i++)
	// 	tmp_input_itemsets[i*max_cols] = -i * penalty;
	// for( int j = 1; j< max_cols ; j++)
    // 	tmp_input_itemsets[j] = -j * penalty;


	// printf("grid_x: %d, grid_y: %d, thread_x: %d, thread_y: %d\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
	//
	size_t membytes = 0;
	membytes += page_align(sizeof(int)*size);
	membytes += page_align(sizeof(int)*size);
	membytes += 309 * 1024 * 1024;

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
	
	long orig_alloc_mem = bemps_begin(tid, block_width, 1, 1, dimBlock.x, dimBlock.y, dimBlock.z, membytes, ret_dev_id);
	
	clock_gettime( CLOCK_REALTIME, &specific_time);
	now = localtime(&specific_time.tv_sec);
	millsec = specific_time.tv_nsec;
	
	millsec = floor (specific_time.tv_nsec/1.0e6);
	
	printf("TID: %d after schedule, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
		now->tm_mon + 1, now->tm_mday, now->tm_hour, 
		now->tm_min, now->tm_sec, millsec);

	if (membytes <= orig_alloc_mem){
		full = 1;
		priority = -5;
	}
	else{
		full = 0;
		priority = 0;
	}

	printf("Full: %d\n", full);

  	printf("ret_dev_id: %d\n", ret_dev_id);

	auto t_start = high_resolution_clock::now();

	CUDA_CHECK(cudaMallocManaged(&referrence_cuda, sizeof(int)*size));
	CUDA_CHECK(cudaMallocManaged(&input_itemsets, sizeof(int)*size));

	auto t_stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Device mem allocation time: " << duration.count() << std::endl;

	ret1.devPtr = referrence_cuda;
	ret1.advice = cudaMemAdviseSetPreferredLocation;
	ret1.device = ret_dev_id;
	ret1.alloc_size = size * sizeof(float);

  	ret2.devPtr = input_itemsets;
	ret2.advice = cudaMemAdviseSetPreferredLocation;
	ret2.device = ret_dev_id;
	ret2.alloc_size = size * sizeof(float);

	mem_list.push_back(ret1);
	mem_list.push_back(ret2);

	if(full == 1){
		for(Parameters var : mem_list){
			// CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, ret.device, 0));
			CUDA_CHECK(cudaMemAdvise(var.devPtr, var.alloc_size, var.advice, var.device));
		}
	}
	
	t_start = high_resolution_clock::now();

	CUfileError_t status;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
	int fd = -1;
	ssize_t ret = -1;

	status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
            std::cerr << "cufile driver open error: " << std::endl;
         // << cuFileGetErrorString(status) << std::endl;
    }

	if(max_rows == 16385){
		printf("Small\n");
		fd = open("small/reference_small.txt", O_RDONLY | O_DIRECT, 0644);
		memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		ret = -1;
 
		ret = cuFileRead(cf_handle, referrence_cuda, sizeof(int)*size, 0, 0);
		printf("!!!!!!!%zd\n", ret);
		cuFileHandleDeregister(cf_handle);
     	close(fd);
		//
		fd = open("small/matrix_small.txt", O_RDONLY | O_DIRECT, 0644);
		memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		ret = -1;
  
		ret = cuFileRead(cf_handle, input_itemsets, sizeof(int)*size, 0, 0);
		printf("!!!!!!!%zd\n", ret);
		cuFileHandleDeregister(cf_handle);
		close(fd);
	}
	else if(max_rows == 32769){
		printf("Large\n");
		fd = open("large/reference_large.txt", O_RDONLY | O_DIRECT, 0644);
		memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		ret = -1;
 
		ret = cuFileRead(cf_handle, referrence_cuda, sizeof(int)*size, 0, 0);
		printf("!!!!!!!%zd\n", ret);
		cuFileHandleDeregister(cf_handle);
     	close(fd);
		//
		fd = open("large/matrix_large.txt", O_RDONLY | O_DIRECT, 0644);
		memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		ret = -1;
  
		ret = cuFileRead(cf_handle, input_itemsets, sizeof(int)*size, 0, 0);
		printf("!!!!!!!%zd\n", ret);
		cuFileHandleDeregister(cf_handle);
		close(fd);
	}

	// For reducing host initilization time in task

	// memcpy(referrence_cuda, tmp_referrence, sizeof(int) * size);
	// memcpy(input_itemsets, tmp_input_itemsets, sizeof(int) * size);

	//

	// if (!input_itemsets)
	// 	fprintf(stderr, "error: can not allocate memory");

    // srand ( 7 );
	
    // for (int i = 0 ; i < max_cols; i++){
	// 	for (int j = 0 ; j < max_rows; j++){
	// 		input_itemsets[i*max_cols+j] = 0;
	// 	}
	// }
	
	// printf("Start Needleman-Wunsch\n");
	
	// for( int i=1; i< max_rows ; i++){    //please define your own sequence. 
    //    input_itemsets[i*max_cols] = rand() % 10 + 1;
	// }
    // for( int j=1; j< max_cols ; j++){    //please define your own sequence.
    //    input_itemsets[j] = rand() % 10 + 1;
	// }


	// for (int i = 1 ; i < max_cols; i++){
	// 	for (int j = 1 ; j < max_rows; j++){
	// 	referrence_cuda[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
	// 	}
	// }

    // for( int i = 1; i< max_rows ; i++)
    //    input_itemsets[i*max_cols] = -i * penalty;
	// for( int j = 1; j< max_cols ; j++)
    //    input_itemsets[j] = -j * penalty;

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Storage to Device memcpy time: " << duration.count() << std::endl;

	printf("Processing top-left matrix\n");
	//process top-left matrix

	// t_start = high_resolution_clock::now();

	// if(full == 1){
	// 	for(Parameters ret : mem_list){
	// 		// CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, ret.device, 0));
	// 		CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
	// 	}
	// }

	// t_stop = high_resolution_clock::now();
	// duration = duration_cast<milliseconds>(t_stop - t_start);
	// std::cout << "Tid: " << tid << " Host to Device memcpy time: " << duration.count() << std::endl;

	t_start = high_resolution_clock::now();

	cudaStream_t s1;
	CUDA_CHECK(cudaStreamCreateWithPriority(&s1, 0, priority));

	cudaEvent_t event1;
	CUDA_CHECK(cudaEventCreateWithFlags(&event1, cudaEventDisableTiming));

	for( int i = 1 ; i <= block_width ; i++){
		dimGrid.x = i;
		dimGrid.y = 1;
		needle_cuda_shared_1<<<dimGrid, dimBlock, 0, s1>>>(referrence_cuda, input_itemsets
		                                      ,max_cols, penalty, i, block_width);
		
		CUDA_CHECK(cudaEventRecord(event1, s1));

		task_monitoring(event1, tid, orig_alloc_mem, membytes);

		// CUDA_CHECK(cudaDeviceSynchronize());
	}

	CUDA_CHECK(cudaStreamSynchronize(s1));
									
	CUDA_CHECK(cudaStreamDestroy(s1));
	CUDA_CHECK(cudaEventDestroy(event1));

	printf("Processing bottom-right matrix\n");

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "First Kernel execution time: " << duration.count() << std::endl;
	
	t_start = high_resolution_clock::now();

	cudaStream_t s2;
	CUDA_CHECK(cudaStreamCreateWithPriority(&s2, 0, priority));

	cudaEvent_t event2;
	CUDA_CHECK(cudaEventCreateWithFlags(&event2, cudaEventDisableTiming));

    //process bottom-right matrix
	for( int i = block_width - 1  ; i >= 1 ; i--){
		dimGrid.x = i;
		dimGrid.y = 1;
		needle_cuda_shared_2<<<dimGrid, dimBlock, 0, s2>>>(referrence_cuda, input_itemsets
		                                      ,max_cols, penalty, i, block_width);

		CUDA_CHECK(cudaEventRecord(event2, s2));

		task_monitoring(event2, tid, orig_alloc_mem, membytes);

		// CUDA_CHECK(cudaDeviceSynchronize());
	}

	CUDA_CHECK(cudaStreamSynchronize(s2));
									
	CUDA_CHECK(cudaStreamDestroy(s2));
	CUDA_CHECK(cudaEventDestroy(event2));

	// CUDA_CHECK(cudaDeviceSynchronize());

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Second kernel execution time: " << duration.count() << std::endl;
    // cudaMemcpy(output_itemsets, matrix_cuda, sizeof(int) * size, cudaMemcpyDeviceToHost);
	
	size_t free_mem, total_mem;
// 	CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

//   printf("Free: %zd, Total: %zd\n", free_mem, total_mem);
  
  CUDA_CHECK(cudaFree(referrence_cuda));

//   CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

//   printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

  if(full == 1){
    ef_mem = page_align(sizeof(int) * size);
    pre_bemps_free(tid, ef_mem);
    ef_cnt = 1;
  }

	t_start = high_resolution_clock::now();

	char* str1 = argv[3];
	char* str2 = "_output.txt";
	strcat(str1, str2);

	fd = open(str1, O_CREAT | O_RDWR | O_DIRECT, 0664);
	if (fd < 0) {
		std::cerr << "file open error:" << std::endl;
	}
	memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = fd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
	status = cuFileHandleRegister(&cf_handle, &cf_descr);
	if (status.err != CU_FILE_SUCCESS) {
		std::cerr << "file register error:" << std::endl;
		close(fd);
	}
	ret = -1;

	ret = cuFileWrite(cf_handle, ret2.devPtr, ret2.alloc_size, 0, 0);
	if (ret < 0)
		printf("Error!!\n");
	else {
		std::cout << "written bytes :" << ret << std::endl;
		ret = 0;
	}
	cuFileHandleDeregister(cf_handle);
	close(fd);
	cuFileDriverClose();

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Device to Storage memcpy time: " << duration.count() << std::endl;

//#define TRACEBACK
#ifdef TRACEBACK
	
FILE *fpo = fopen("result.txt","w");
fprintf(fpo, "print traceback value GPU:\n");

// CUDA_CHECK(cudaMemPrefetchAsync(ret1.devPtr, ret1.alloc_size, cudaCpuDeviceId, 0))
// CUDA_CHECK(cudaMemPrefetchAsync(ret2.devPtr, ret2.alloc_size, cudaCpuDeviceId, 0))
// CUDA_CHECK(cudaDeviceSynchronize());

for (int i = max_rows - 2,  j = max_rows - 2; i>=0, j>=0;){
	int nw, n, w, traceback;
	if ( i == max_rows - 2 && j == max_rows - 2 )
		fprintf(fpo, "%d ", input_itemsets[ i * max_cols + j]); //print the first element
	if ( i == 0 && j == 0 )
	   break;
	if ( i > 0 && j > 0 ){
		nw = input_itemsets[(i - 1) * max_cols + j - 1];
		w  = input_itemsets[ i * max_cols + j - 1 ];
		n  = input_itemsets[(i - 1) * max_cols + j];
	}
	else if ( i == 0 ){
		nw = n = LIMIT;
		w  = input_itemsets[ i * max_cols + j - 1 ];
	}
	else if ( j == 0 ){
		nw = w = LIMIT;
		n  = input_itemsets[(i - 1) * max_cols + j];
	}
	else{
	}

	//traceback = maximum(nw, w, n);
	int new_nw, new_w, new_n;
	new_nw = nw + referrence_cuda[i * max_cols + j];
	new_w = w - penalty;
	new_n = n - penalty;
	
	traceback = maximum(new_nw, new_w, new_n);
	if(traceback == new_nw)
		traceback = nw;
	if(traceback == new_w)
		traceback = w;
	if(traceback == new_n)
		traceback = n;
		
	fprintf(fpo, "%d ", traceback);

	if(traceback == nw )
	{i--; j--; continue;}

	else if(traceback == w )
	{j--; continue;}

	else if(traceback == n )
	{i--; continue;}

	else
	;
}

fclose(fpo);

#endif
	t_start = high_resolution_clock::now();

	cudaFree(input_itemsets);

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Device memory deallocation time: " << duration.count() << std::endl;

	bemps_free(tid);

	clock_gettime( CLOCK_REALTIME, &specific_time);
	now = localtime(&specific_time.tv_sec);
	millsec = specific_time.tv_nsec;

	millsec = floor (specific_time.tv_nsec/1.0e6);

	printf("TID: %d finish work, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
        now->tm_mon + 1, now->tm_mday, now->tm_hour, 
        now->tm_min, now->tm_sec, millsec);

	// free(referrence);
	// free(input_itemsets);
	// free(output_itemsets);
	
}

