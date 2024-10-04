//====================================================================================================100
//		UPDATE
//====================================================================================================100

//    2006.03   Rob Janiczek
//        --creation of prototype version
//    2006.03   Drew Gilliam
//        --rewriting of prototype version into current version
//        --got rid of multiple function calls, all code in a  
//         single function (for speed)
//        --code cleanup & commenting
//        --code optimization efforts   
//    2006.04   Drew Gilliam
//        --added diffusion coefficent saturation on [0,1]
//		2009.12 Lukasz G. Szafaryn
//		-- reading from image, command line inputs
//		2010.01 Lukasz G. Szafaryn
//		--comments

//====================================================================================================100
//	DEFINE / INCLUDE
//====================================================================================================100

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <string>
#include <bitset>
#include <bemps.hpp>
#include <chrono>
#include <iostream>

#include "define.c"
#include "extract_kernel.cu"
#include "prepare_kernel.cu"
#include "reduce_kernel.cu"
#include "srad_kernel.cu"
#include "srad2_kernel.cu"
#include "compress_kernel.cu"
#include "graphics.c"
#include "resize.c"
#include "timer.c"

#include "cufile.h"
#include <unistd.h>
#include <error.h>
#include <fcntl.h>
#include <errno.h>

#include "device.c"				// (in library path specified to compiler)	needed by for device functions

using namespace std::chrono;

#define page_size 4096
#define VA_block 2097152

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

void init (fp* arr, int size){
	for(int i = 0; i < size / sizeof(fp); i++){
		arr[i] = 0;
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
			// cudaStream_t s_e;
			// CUDA_CHECK(cudaStreamCreate(&s_e));
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
					int loop = 0;
					for(Parameters ret : mem_list){
						CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
						if(loop < 5){
							CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, ret.device, s_e));
						}
						loop++;
					}
					CUDA_CHECK(cudaStreamSynchronize(s_e));
					CUDA_CHECK(cudaStreamDestroy(s_e));
					break;
				}
			}
			// CUDA_CHECK(cudaStreamDestroy(s_e));
		}
	}
}

//====================================================================================================100
//	MAIN FUNCTION
//====================================================================================================100

int main(int argc, char *argv []){

	//================================================================================80
	// 	VARIABLES
	//================================================================================80
	int tid = atoi(argv[5]);
	
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

	size_t ef_mem;
	int ef_cnt = 0;

	int ret_dev_id;

	Parameters ret1;
	Parameters ret2;
	Parameters ret3;
	Parameters ret4;
	Parameters ret5;
	Parameters ret6;
	Parameters ret7;
	Parameters ret8;
	Parameters ret9;
	Parameters ret10;
	Parameters ret11;
	Parameters ret12;
	Parameters ret13;
	
	// time
	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long time6;
	long long time7;
	long long time8;
	long long time9;
	long long time10;
	long long time11;
	long long time12;

	time0 = get_time();

	fp* tmp_image_ori, *tmp_image;

    // inputs image, input paramenters
    fp* image_ori;																// originalinput image
	int image_ori_rows;
	int image_ori_cols;
	long image_ori_elem;

    // inputs image, input paramenters
    fp* image;															// input image
    int Nr,Nc;													// IMAGE nbr of rows/cols/elements
	long Ne;

	// algorithm parameters
    int niter;																// nbr of iterations
    fp lambda;															// update step size

    // size of IMAGE
	int r1,r2,c1,c2;												// row/col coordinates of uniform ROI
	long NeROI;														// ROI nbr of elements

    // surrounding pixel indicies
    int *iN,*iS,*jE,*jW;    

    // counters
    int iter;   // primary loop
    long i,j;    // image row/col

	// memory sizes
	int mem_size_i;
	int mem_size_j;
	int mem_size_single;

	//================================================================================80
	// 	GPU VARIABLES
	//================================================================================80

	// CUDA kernel execution parameters
	dim3 threads;
	int blocks_x;
	dim3 blocks;
	dim3 blocks2;
	dim3 blocks3;

	// memory sizes
	int mem_size;															// matrix memory size

	// HOST
	int no;
	int mul;
	fp total;
	fp total2;
	fp meanROI;
	fp meanROI2;
	fp varROI;
	fp q0sqr;

	// DEVICE
	fp* sums;															// partial sum
	fp* sums2;
	int* tmp_iN;
	int* tmp_iS;
	int* tmp_jE;
	int* tmp_jW;
	fp* dN; 
	fp* dS; 
	fp* dW; 
	fp* dE;
	fp* I;																// input IMAGE on DEVICE
	fp* c;

	time1 = get_time();

	//================================================================================80
	// 	GET INPUT PARAMETERS
	//================================================================================80

	if(argc != 6){
		printf("ERROR: wrong number of arguments\n");
		return 0;
	}
	else{
		niter = atoi(argv[1]);
		lambda = atof(argv[2]);
		Nr = atoi(argv[3]);						// it is 502 in the original image
		Nc = atoi(argv[4]);						// it is 458 in the original image
	}

	time2 = get_time();

	//================================================================================80
	// 	READ IMAGE (SIZE OF IMAGE HAS TO BE KNOWN)
	//================================================================================80
	
	Ne = Nr*Nc;

    // read image
	image_ori_rows = 502;
	image_ori_cols = 458;
	image_ori_elem = image_ori_rows * image_ori_cols;

	tmp_image_ori = (fp*)malloc(sizeof(fp) * image_ori_elem);
	tmp_image = (fp*)malloc(sizeof(fp) * Ne);

	read_graphics(	"../../../data/srad/image.pgm",
								tmp_image_ori,
								image_ori_rows,
								image_ori_cols,
								1);

	// resize(	tmp_image_ori,
	// 	image_ori_rows,
	// 	image_ori_cols,
	// 	tmp_image,
	// 	Nr,
	// 	Nc,
	// 	1);

	r1     = 0;											// top row index of ROI
    r2     = Nr - 1;									// bottom row index of ROI
    c1     = 0;											// left column index of ROI
    c2     = Nc - 1;									// right column index of ROI

	// ROI image size
	NeROI = (r2-r1+1)*(c2-c1+1);											// number of elements in ROI, ROI size

	// allocate variables for surrounding pixels
	mem_size = sizeof(fp) * Ne;
	mem_size_i = sizeof(int) * Nr;											//
	mem_size_j = sizeof(int) * Nc;

	size_t membytes = 0;
	membytes += page_align(mem_size);
	membytes += page_align(mem_size_i);
	// membytes += page_align(mem_size_i);
	// membytes += page_align(mem_size_j);
	// membytes += page_align(mem_size_j);
	membytes += page_align(mem_size);
	membytes += page_align(mem_size);
	membytes += page_align(mem_size);
	membytes += page_align(mem_size);
	membytes += page_align(mem_size);
	membytes += page_align(mem_size);
	membytes += page_align(mem_size);
  	membytes += 309 * 1024 * 1024;

	threads.x = NUMBER_THREADS;												// define the number of threads in the block
	threads.y = 1;
	blocks_x = Ne/threads.x;
	if (Ne % threads.x != 0){												// compensate for division remainder above by adding one grid
		blocks_x = blocks_x + 1;																	
	}
	blocks.x = blocks_x;													// define the number of blocks in the grid
	blocks.y = 1;

	// printf("blocks_x: %d, blocks_y: %d, threads_x: %d, threads_y:%d\n", blocks.x, blocks.y, threads.x, threads.y);

    clock_gettime( CLOCK_REALTIME, &specific_time);
    now = localtime(&specific_time.tv_sec);
    millsec = specific_time.tv_nsec;

    millsec = floor (specific_time.tv_nsec/1.0e6);

    printf("TID: %d before schedule, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
		now->tm_mon + 1, now->tm_mday, now->tm_hour, 
		now->tm_min, now->tm_sec, millsec);

	long orig_alloc_mem = bemps_begin(tid, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z, membytes, ret_dev_id);

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

	// int low, max;

	// CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&low, &max));

	// printf("Low: %d, Max: %d\n", low, max);

	// int* dummy;
	// size_t d_size = 13127122944;
	// CUDA_CHECK(cudaMalloc(&dummy, d_size));
	// CUDA_CHECK(cudaMemset(dummy, 0, d_size));

	// t_start = high_resolution_clock::now();

	// cudaMallocManaged(&image_ori, sizeof(fp) * image_ori_elem);

	// t_stop = high_resolution_clock::now();
	// duration = duration_cast<milliseconds>(t_stop - t_start);
	// std::cout << "Tid: " << tid << " First device mem allocation time: " << duration.count() << std::endl;

	time3 = get_time();

	//================================================================================80
	// 	KERNEL EXECUTION PARAMETERS
	//================================================================================80

	time4 = get_time();

	//================================================================================80
	// 	RESIZE IMAGE (ASSUMING COLUMN MAJOR STORAGE OF image_orig)
	//================================================================================80

	// image = (fp*)malloc(sizeof(fp) * Ne);

	auto t_start = high_resolution_clock::now();

	CUDA_CHECK(cudaMallocManaged(&image, sizeof(fp) * Ne));

	CUDA_CHECK(cudaMallocManaged(&iN, mem_size_i));
	CUDA_CHECK(cudaMallocManaged(&iS, mem_size_i));

	CUDA_CHECK(cudaMallocManaged(&jW, mem_size_j));
	CUDA_CHECK(cudaMallocManaged(&jE, mem_size_j));

	CUDA_CHECK(cudaMallocManaged(&sums, mem_size));
	CUDA_CHECK(cudaMallocManaged(&sums2, mem_size));
	CUDA_CHECK(cudaMallocManaged(&dN, mem_size));
	CUDA_CHECK(cudaMallocManaged(&dS, mem_size));
	CUDA_CHECK(cudaMallocManaged(&dW, mem_size));
	CUDA_CHECK(cudaMallocManaged(&dE, mem_size));
	CUDA_CHECK(cudaMallocManaged(&c, mem_size));

	auto t_stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Device mem allocation time: " << duration.count() << std::endl;

	// resize(	image_ori,
	// 			image_ori_rows,
	// 			image_ori_cols,
	// 			image,
	// 			Nr,
	// 			Nc,
	// 			1);

	ret1.devPtr = image;
	ret1.advice = cudaMemAdviseSetPreferredLocation;
	ret1.device = ret_dev_id;
	ret1.alloc_size = mem_size;

	ret2.devPtr = iN;
	ret2.advice = cudaMemAdviseSetPreferredLocation;
	ret2.device = ret_dev_id;
	ret2.alloc_size = mem_size_i;

	ret3.devPtr = iS;
	ret3.advice = cudaMemAdviseSetPreferredLocation;
	ret3.device = ret_dev_id;
	ret3.alloc_size = mem_size_i;
	
	ret4.devPtr = jW;
	ret4.advice = cudaMemAdviseSetPreferredLocation;
	ret4.device = ret_dev_id;
	ret4.alloc_size = mem_size_j;

	ret5.devPtr = jE;
	ret5.advice = cudaMemAdviseSetPreferredLocation;
	ret5.device = ret_dev_id;
	ret5.alloc_size = mem_size_j;

	ret6.devPtr = sums;
	ret6.advice = cudaMemAdviseSetPreferredLocation;
	ret6.device = ret_dev_id;
	ret6.alloc_size = mem_size;

	ret7.devPtr = sums2;
	ret7.advice = cudaMemAdviseSetPreferredLocation;
	ret7.device = ret_dev_id;
	ret7.alloc_size = mem_size;
	
	ret8.devPtr = dN;
	ret8.advice = cudaMemAdviseSetPreferredLocation;
	ret8.device = ret_dev_id;
	ret8.alloc_size = mem_size;

	ret9.devPtr = dS;
	ret9.advice = cudaMemAdviseSetPreferredLocation;
	ret9.device = ret_dev_id;
	ret9.alloc_size = mem_size;

	ret10.devPtr = dW;
	ret10.advice = cudaMemAdviseSetPreferredLocation;
	ret10.device = ret_dev_id;
	ret10.alloc_size = mem_size;

	ret11.devPtr = dE;
	ret11.advice = cudaMemAdviseSetPreferredLocation;
	ret11.device = ret_dev_id;
	ret11.alloc_size = mem_size;

	ret12.devPtr = c;
	ret12.advice = cudaMemAdviseSetPreferredLocation;
	ret12.device = ret_dev_id;
	ret12.alloc_size = mem_size;

	mem_list.push_back(ret1);
	mem_list.push_back(ret2);
	mem_list.push_back(ret3);
	mem_list.push_back(ret4);
	mem_list.push_back(ret5);
	mem_list.push_back(ret6);
	mem_list.push_back(ret7);
	mem_list.push_back(ret8);
	mem_list.push_back(ret9);
	mem_list.push_back(ret10);
	mem_list.push_back(ret11);
	mem_list.push_back(ret12);

	time5 = get_time();

	//================================================================================80
	// 	SETUP
	//================================================================================80

	t_start = high_resolution_clock::now();

	int fd = -1;
	ssize_t ret = -1;

	if(Nr == 11000){
		printf("Small\n");
		fd = open("small/iN_small.txt", O_RDONLY, 0644);
		printf("%d\n", fd);

		ret = -1;
		ret = read(fd, iN, mem_size_i);
		printf("%zd\n", ret);
		close(fd);

		fd = open("small/iS_small.txt", O_RDONLY, 0644);
		printf("%d\n", fd);

		ret = -1;
		ret = read(fd, iS, mem_size_i);
		printf("%zd\n", ret);
		close(fd);

		fd = open("small/jW_small.txt", O_RDONLY, 0644);
		printf("%d\n", fd);

		ret = -1;
		ret = read(fd, jW, mem_size_j);
		printf("%zd\n", ret);
		close(fd);

		fd = open("small/jE_small.txt", O_RDONLY, 0644);
		printf("%d\n", fd);

		ret = -1;
		ret = read(fd, jE, mem_size_j);
		printf("%zd\n", ret);
		close(fd);
	}
	else if(Nr == 15000){
		printf("Large1\n");
		fd = open("large1/iN_large1.txt", O_RDONLY, 0644);
		printf("%d\n", fd);

		ret = -1;
		ret = read(fd, iN, mem_size_i);
		printf("%zd\n", ret);
		close(fd);

		fd = open("large1/iS_large1.txt", O_RDONLY, 0644);
		printf("%d\n", fd);

		ret = -1;
		ret = read(fd, iS, mem_size_i);
		printf("%zd\n", ret);
		close(fd);

		fd = open("large1/jW_large1.txt", O_RDONLY, 0644);
		printf("%d\n", fd);

		ret = -1;
		ret = read(fd, jW, mem_size_j);
		printf("%zd\n", ret);
		close(fd);

		fd = open("large1/jE_large1.txt", O_RDONLY, 0644);
		printf("%d\n", fd);

		ret = -1;
		ret = read(fd, jE, mem_size_j);
		printf("%zd\n", ret);
		close(fd);
	}
	else if(Nr = 20000){
		printf("Large2\n");
		fd = open("large2/iN_large2.txt", O_RDONLY, 0644);
		printf("%d\n", fd);

		ret = -1;
		ret = read(fd, iN, mem_size_i);
		printf("%zd\n", ret);
		close(fd);

		fd = open("large2/iS_large2.txt", O_RDONLY, 0644);
		printf("%d\n", fd);

		ret = -1;
		ret = read(fd, iS, mem_size_i);
		printf("%zd\n", ret);
		close(fd);

		fd = open("large2/jW_large2.txt", O_RDONLY, 0644);
		printf("%d\n", fd);

		ret = -1;
		ret = read(fd, jW, mem_size_j);
		printf("%zd\n", ret);
		close(fd);

		fd = open("large2/jE_large2.txt", O_RDONLY, 0644);
		printf("%d\n", fd);

		ret = -1;
		ret = read(fd, jE, mem_size_j);
		printf("%zd\n", ret);
		close(fd);
	}

	resize(	tmp_image_ori,
		image_ori_rows,
		image_ori_cols,
		image,
		Nr,
		Nc,
		1);

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Host initialization time: " << duration.count() << std::endl;

	t_start = high_resolution_clock::now();

	if(full == 1){
		for(Parameters var : mem_list){
			// if(loop < 5){
			// 	CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, ret.device, 0));
			// }
			CUDA_CHECK(cudaMemAdvise(var.devPtr, var.alloc_size, var.advice, var.device));
		}
		CUDA_CHECK(cudaMemPrefetchAsync(image, mem_size, ret_dev_id, 0));
		CUDA_CHECK(cudaMemPrefetchAsync(iN, mem_size_i, ret_dev_id, 0));
		CUDA_CHECK(cudaMemPrefetchAsync(iS, mem_size_i, ret_dev_id, 0));
		CUDA_CHECK(cudaMemPrefetchAsync(jW, mem_size_j, ret_dev_id, 0));
		CUDA_CHECK(cudaMemPrefetchAsync(jE, mem_size_j, ret_dev_id, 0));
	}

	// size_t free_mem, total_mem;
	// CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

	// printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Host to Device memcpy time: " << duration.count() << std::endl;

	t_start = high_resolution_clock::now();

	// if(full == 1){
	// 	int loop = 0;
	// 	printf("Hi\n");
	// 	for(Parameters ret : mem_list){
	// 		// if(loop < 5){
	// 		// 	CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, ret.device, 0));
	// 		// }
	// 		CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
	// 		loop++;
	// 	}
	// }

	// CUDA_CHECK(cudaDeviceSynchronize());

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Mem advise time: " << duration.count() << std::endl;

	//================================================================================80
	// 	GPU SETUP
	//================================================================================80

	// cudaMemPrefetchAsync(iN, mem_size_i, 0, 0);
	// cudaMemPrefetchAsync(iS, mem_size_i, 0, 0);
	// cudaMemPrefetchAsync(jE, mem_size_j, 0, 0);
	// cudaMemPrefetchAsync(jW, mem_size_j, 0, 0);

	// checkCUDAError("setup");

	//================================================================================80
	// 	COPY INPUT TO CPU
	//================================================================================80

	// cudaMemPrefetchAsync(image, mem_size, ret_dev_id, 0);

	time6 = get_time();

	if(full == 0){
		clock_gettime( CLOCK_REALTIME, &specific_time);
		now = localtime(&specific_time.tv_sec);
		millsec = specific_time.tv_nsec;

		millsec = floor (specific_time.tv_nsec/1.0e6);

		printf("TID: %d waiting start, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
			now->tm_mon + 1, now->tm_mday, now->tm_hour, 
			now->tm_min, now->tm_sec, millsec);

		el_wait(tid);

		clock_gettime( CLOCK_REALTIME, &specific_time);
		now = localtime(&specific_time.tv_sec);
		millsec = specific_time.tv_nsec;

		millsec = floor (specific_time.tv_nsec/1.0e6);

		printf("TID: %d waiting end, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
			now->tm_mon + 1, now->tm_mday, now->tm_hour, 
			now->tm_min, now->tm_sec, millsec);
	}

	//================================================================================80
	// 	SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
	//================================================================================80

	cudaEvent_t e_start, e_stop;
	CUDA_CHECK(cudaEventCreate(&e_start));
	CUDA_CHECK(cudaEventCreate(&e_stop));

	float total_e, time;

	// t_start = high_resolution_clock::now();

	cudaStream_t s1;
	CUDA_CHECK(cudaStreamCreateWithPriority(&s1, 0, priority));

	cudaEvent_t event1;
	CUDA_CHECK(cudaEventCreateWithFlags(&event1, cudaEventDisableTiming));

	CUDA_CHECK(cudaEventRecord(e_start));

	// if(full == 1){
	// 	clock_gettime( CLOCK_REALTIME, &specific_time);
	// 			now = localtime(&specific_time.tv_sec);
	// 			millsec = specific_time.tv_nsec;

	// 			millsec = floor (specific_time.tv_nsec/1.0e6);

	// 			printf("TID: %d sending signal start, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
	// 				now->tm_mon + 1, now->tm_mday, now->tm_hour, 
	// 				now->tm_min, now->tm_sec, millsec);

	// 			nl_signal(tid);

	// 			clock_gettime( CLOCK_REALTIME, &specific_time);
	// 			now = localtime(&specific_time.tv_sec);
	// 			millsec = specific_time.tv_nsec;

	// 			millsec = floor (specific_time.tv_nsec/1.0e6);

	// 			printf("TID: %d sending signal start, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
	// 				now->tm_mon + 1, now->tm_mday, now->tm_hour, 
	// 				now->tm_min, now->tm_sec, millsec);
	// }
	
	extract<<<blocks, threads, 0, s1>>>(	Ne,
									image);

	// extract<<<blocks, threads>>>(	Ne,
	// 	image);

	CUDA_CHECK(cudaEventRecord(event1, s1));

	task_monitoring(event1, tid, orig_alloc_mem, membytes);

	CUDA_CHECK(cudaEventRecord(e_stop));
	CUDA_CHECK(cudaEventSynchronize(e_stop));
	CUDA_CHECK(cudaEventElapsedTime(&total_e, e_start, e_stop));

	// CUDA_CHECK(cudaDeviceSynchronize());

	printf("E_kernel: %f\n", total_e);

	if(full == 1){
		clock_gettime( CLOCK_REALTIME, &specific_time);
				now = localtime(&specific_time.tv_sec);
				millsec = specific_time.tv_nsec;

				millsec = floor (specific_time.tv_nsec/1.0e6);

				printf("TID: %d sending signal start, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
					now->tm_mon + 1, now->tm_mday, now->tm_hour, 
					now->tm_min, now->tm_sec, millsec);

				nl_signal(tid);

				clock_gettime( CLOCK_REALTIME, &specific_time);
				now = localtime(&specific_time.tv_sec);
				millsec = specific_time.tv_nsec;

				millsec = floor (specific_time.tv_nsec/1.0e6);

				printf("TID: %d sending signal start, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
					now->tm_mon + 1, now->tm_mday, now->tm_hour, 
					now->tm_min, now->tm_sec, millsec);
	}
	
	// t_stop = high_resolution_clock::now();
	// duration = duration_cast<milliseconds>(t_stop - t_start);
	// std::cout << "Tid: " << tid << " Kernel execution time: " << duration.count() << std::endl;

	// checkCUDAError("extract");

	time7 = get_time();

	//================================================================================80
	// 	COMPUTATION
	//================================================================================80

	cudaEvent_t *p_start, *p_stop, *r_start, *r_stop, *s1_start, *s1_stop, *s2_start, *s2_stop, *cm_start, *cm_stop;

	// CUDA_CHECK(cudaMallocManaged((void **)&p_start, sizeof(cudaEvent_t) * 100))
	// CUDA_CHECK(cudaMallocManaged((void **)&p_stop, sizeof(cudaEvent_t) * 100))

	p_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 100);
	p_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 100);
	r_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 400);
	r_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 400);
	s1_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 100);
	s1_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 100);
	s2_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 100);
	s2_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 100);
	cm_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 100);
	cm_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 100);

	// CUDA_CHECK(cudaEventCreate(p_start));
	// CUDA_CHECK(cudaEventCreate(p_stop));
	// cudaEventCreate(&r_start);
	// cudaEventCreate(&r_stop);
	// cudaEventCreate(&s1_start);
	// cudaEventCreate(&s1_stop);
	// cudaEventCreate(&s2_start);
	// cudaEventCreate(&s2_stop);

	float total_p, total_r, total_s1, total_s2, memcpy_time;
	total_p = 0;
	total_r = 0;
	total_s1 = 0;
	total_s2 = 0;
	memcpy_time = 0;

	// printf("iterations: ");
	cudaStream_t s2, s3;
	cudaEvent_t event2, event3;
	// execute main loop

	t_start = high_resolution_clock::now();

	for (iter=0; iter<niter; iter++){										// do for the number of iterations input parameter
		int r_iter = 0;
	// printf("%d ", iter);
	// fflush(NULL);

		// if(iter == 0){
		// 	CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

		// 	printf("Free: %zd, Total: %zd\n", free_mem, total_mem);
		// }

		CUDA_CHECK(cudaStreamCreateWithPriority(&s2, 0, priority));

		CUDA_CHECK(cudaEventCreateWithFlags(&event2, cudaEventDisableTiming));

		CUDA_CHECK(cudaEventCreate(&p_start[iter]));
		CUDA_CHECK(cudaEventCreate(&p_stop[iter]));
		CUDA_CHECK(cudaEventRecord(p_start[iter]));

		// execute square kernel
		prepare<<<blocks, threads, 0, s2>>>(	Ne,
										image,
										sums,
										sums2);

		// prepare<<<blocks, threads>>>(	Ne,
		// 	image,
		// 	sums,
		// 	sums2);

		CUDA_CHECK(cudaEventRecord(event2, s2));

		task_monitoring(event2, tid, orig_alloc_mem, membytes);
								
		CUDA_CHECK(cudaEventRecord(p_stop[iter]));
		// cudaEventSynchronize(p_stop);
		// cudaEventElapsedTime(&time, p_start, p_stop);
		// total_p += time;

		// CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaStreamSynchronize(s2));

		CUDA_CHECK(cudaStreamDestroy(s2));
		CUDA_CHECK(cudaEventDestroy(event2));

		// checkCUDAError("prepare");

		// performs subsequent reductions of sums
		blocks2.x = blocks.x;												// original number of blocks
		blocks2.y = blocks.y;												
		no = Ne;														// original number of sum elements
		mul = 1;														// original multiplier
		
		// CUDA_CHECK(cudaEventCreate(&r_start[iter]));
		// CUDA_CHECK(cudaEventCreate(&r_stop[iter]));
		// CUDA_CHECK(cudaEventRecord(r_start[iter]));

		// if(iter == 0){
		// 	CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

		// 	printf("Free: %zd, Total: %zd\n", free_mem, total_mem);
		// }

		while(blocks2.x != 0){
			// printf("Hello\n");
			checkCUDAError("before reduce");

			CUDA_CHECK(cudaStreamCreateWithPriority(&s3, 0, priority));

			CUDA_CHECK(cudaEventCreateWithFlags(&event3, cudaEventDisableTiming));

			CUDA_CHECK(cudaEventCreate(&r_start[iter*4+r_iter]));
			CUDA_CHECK(cudaEventCreate(&r_stop[iter*4+r_iter]));
			CUDA_CHECK(cudaEventRecord(r_start[iter*4+r_iter]));


			// cudaEventRecord(r_start);

			// run kernel
			reduce<<<blocks2, threads, 0, s3>>>(	Ne,
											no,
											mul,
											sums, 
											sums2);

			// reduce<<<blocks2, threads>>>(	Ne,
			// 	no,
			// 	mul,
			// 	sums, 
			// 	sums2);

			CUDA_CHECK(cudaEventRecord(event3, s3));

			task_monitoring(event3, tid, orig_alloc_mem, membytes);
			
			CUDA_CHECK(cudaEventRecord(r_stop[iter*4+r_iter]));

			// cudaEventRecord(r_stop);
			// cudaEventSynchronize(r_stop);
			// cudaEventElapsedTime(&time, r_start, r_stop);
			// total_r += time;

			// CUDA_CHECK(cudaDeviceSynchronize());
			CUDA_CHECK(cudaStreamSynchronize(s3));
									
			CUDA_CHECK(cudaStreamDestroy(s3));
			CUDA_CHECK(cudaEventDestroy(event3));

			// checkCUDAError("reduce");

			// update execution parameters
			no = blocks2.x;												// get current number of elements
			if(blocks2.x == 1){
				blocks2.x = 0;
			}
			else{
				mul = mul * NUMBER_THREADS;									// update the increment
				blocks_x = blocks2.x/threads.x;								// number of blocks
				if (blocks2.x % threads.x != 0){							// compensate for division remainder above by adding one grid
					blocks_x = blocks_x + 1;
				}
				blocks2.x = blocks_x;
				blocks2.y = 1;
			}

			checkCUDAError("after reduce");
			
			r_iter++;
		}

		// CUDA_CHECK(cudaEventRecord(r_stop[iter]));

		checkCUDAError("before copy sum");

		CUDA_CHECK(cudaEventCreate(&cm_start[iter]));
		CUDA_CHECK(cudaEventCreate(&cm_stop[iter]));
		CUDA_CHECK(cudaEventRecord(cm_start[iter]));
		
		// copy total sums to device
		mem_size_single = sizeof(fp) * 1;
		// cudaMemcpy(&total, sums, mem_size_single, cudaMemcpyDeviceToHost);
		total = sums[0];
		// CUDA_CHECK(cudaMemPrefetchAsync(sums, mem_size_single, cudaCpuDeviceId, 0));
		// cudaMemcpy(&total2, sums2, mem_size_single, cudaMemcpyDeviceToHost);
		total2 = sums2[0];
		// CUDA_CHECK(cudaMemPrefetchAsync(sums2, mem_size_single, cudaCpuDeviceId, 0));

		checkCUDAError("copy sum");

		CUDA_CHECK(cudaEventRecord(cm_stop[iter]));

		// auto cp_start = high_resolution_clock::now();

		// calculate statistics
		meanROI	= total / fp(NeROI);										// gets mean (average) value of element in ROI
		meanROI2 = meanROI * meanROI;										//
		varROI = (total2 / fp(NeROI)) - meanROI2;						// gets variance of ROI								
		q0sqr = varROI / meanROI2;											// gets standard deviation of ROI

		// auto cp_stop = high_resolution_clock::now();
		// auto cp_duration = duration_cast<milliseconds>(cp_stop - cp_start);
		// std::cout << "Host Processing: " << cp_duration.count() << std::endl;
		// memcpy_time += cp_duration.count();

		cudaStream_t s4;
		CUDA_CHECK(cudaStreamCreateWithPriority(&s4, 0, priority));

		cudaEvent_t event4;
		CUDA_CHECK(cudaEventCreateWithFlags(&event4, cudaEventDisableTiming));

		CUDA_CHECK(cudaEventCreate(&s1_start[iter]));
		CUDA_CHECK(cudaEventCreate(&s1_stop[iter]));
		CUDA_CHECK(cudaEventRecord(s1_start[iter]));

		// execute srad kernel
		srad<<<blocks, threads, 0, s4>>>(	lambda,									// SRAD coefficient 
									Nr,										// # of rows in input image
									Nc,										// # of columns in input image
									Ne,										// # of elements in input image
									iN,									// indices of North surrounding pixels
									iS,									// indices of South surrounding pixels
									jE,									// indices of East surrounding pixels
									jW,									// indices of West surrounding pixels
									dN,									// North derivative
									dS,									// South derivative
									dW,									// West derivative
									dE,									// East derivative
									q0sqr,									// standard deviation of ROI 
									c,									// diffusion coefficient
									image);									// output image

		// srad<<<blocks, threads>>>(	lambda,									// SRAD coefficient 
		// 	Nr,										// # of rows in input image
		// 	Nc,										// # of columns in input image
		// 	Ne,										// # of elements in input image
		// 	iN,									// indices of North surrounding pixels
		// 	iS,									// indices of South surrounding pixels
		// 	jE,									// indices of East surrounding pixels
		// 	jW,									// indices of West surrounding pixels
		// 	dN,									// North derivative
		// 	dS,									// South derivative
		// 	dW,									// West derivative
		// 	dE,									// East derivative
		// 	q0sqr,									// standard deviation of ROI 
		// 	c,									// diffusion coefficient
		// 	image);									// output image

		CUDA_CHECK(cudaEventRecord(event4, s4));

		task_monitoring(event4, tid, orig_alloc_mem, membytes);
				
		CUDA_CHECK(cudaEventRecord(s1_stop[iter]));
		// cudaEventSynchronize(s1_stop);
		// cudaEventElapsedTime(&time, s1_start, s1_stop);
		// total_s1 += time;

		// CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaStreamSynchronize(s4));

		CUDA_CHECK(cudaStreamDestroy(s4));
		CUDA_CHECK(cudaEventDestroy(event4));

		// checkCUDAError("srad");

		cudaStream_t s5;
		CUDA_CHECK(cudaStreamCreateWithPriority(&s5, 0, priority));

		cudaEvent_t event5;
		CUDA_CHECK(cudaEventCreateWithFlags(&event5, cudaEventDisableTiming));

		CUDA_CHECK(cudaEventCreate(&s2_start[iter]));
		CUDA_CHECK(cudaEventCreate(&s2_stop[iter]));
		CUDA_CHECK(cudaEventRecord(s2_start[iter]));

		// execute srad2 kernel
		srad2<<<blocks, threads, 0, s5>>>(	lambda,									// SRAD coefficient 
									Nr,										// # of rows in input image
									Nc,										// # of columns in input image
									Ne,										// # of elements in input image
									iN,									// indices of North surrounding pixels
									iS,									// indices of South surrounding pixels
									jE,									// indices of East surrounding pixels
									jW,									// indices of West surrounding pixels
									dN,									// North derivative
									dS,									// South derivative
									dW,									// West derivative
									dE,									// East derivative
									c,									// diffusion coefficient
									image);									// output image

		// srad2<<<blocks, threads>>>(	lambda,									// SRAD coefficient 
		// 	Nr,										// # of rows in input image
		// 	Nc,										// # of columns in input image
		// 	Ne,										// # of elements in input image
		// 	iN,									// indices of North surrounding pixels
		// 	iS,									// indices of South surrounding pixels
		// 	jE,									// indices of East surrounding pixels
		// 	jW,									// indices of West surrounding pixels
		// 	dN,									// North derivative
		// 	dS,									// South derivative
		// 	dW,									// West derivative
		// 	dE,									// East derivative
		// 	c,									// diffusion coefficient
		// 	image);									// output image

		CUDA_CHECK(cudaEventRecord(event5, s5));

		task_monitoring(event5, tid, orig_alloc_mem, membytes);
							
		CUDA_CHECK(cudaEventRecord(s2_stop[iter]));
		// cudaEventSynchronize(s2_stop);
		// cudaEventElapsedTime(&time, s2_start, s2_stop);
		// total_s2 += time;

		// CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaStreamSynchronize(s5));
		
		CUDA_CHECK(cudaStreamDestroy(s5));
		CUDA_CHECK(cudaEventDestroy(event5));

		// checkCUDAError("srad2");

	}

	CUDA_CHECK(cudaEventSynchronize(p_stop[99]));
	CUDA_CHECK(cudaEventSynchronize(r_stop[399]));
	CUDA_CHECK(cudaEventSynchronize(s1_stop[99]));
	CUDA_CHECK(cudaEventSynchronize(s2_stop[99]));

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Kernel execution time: " << duration.count() << std::endl;

	// CUDA_CHECK(cudaEventSynchronize(p_stop[99]));
	// CUDA_CHECK(cudaEventSynchronize(r_stop[99]));
	// CUDA_CHECK(cudaEventSynchronize(s1_stop[99]));
	// CUDA_CHECK(cudaEventSynchronize(s2_stop[99]));

	for(int i = 0; i < 100; i++){
		cudaEventElapsedTime(&time, p_start[i], p_stop[i]);
		// printf("%f ", time);
		total_p += time;
		for(int j = 0; j < 4; j++){
			cudaEventElapsedTime(&time, r_start[i*4+j], r_stop[i*4+j]);
			// printf("%f ", time);
			total_r += time;
		}
		cudaEventElapsedTime(&time, cm_start[i], cm_stop[i]);
		// printf("%f ", time);
		memcpy_time += time;
		cudaEventElapsedTime(&time, s1_start[i], s1_stop[i]);
		// printf("%f ", time);
		total_s1 += time;
		cudaEventElapsedTime(&time, s2_start[i], s2_stop[i]);
		// printf("%f\n", time);
		total_s2 += time;
	}
	


	printf("P_kernel: %f, R_kernel: %f, Memcpy: %f, S1_kernel: %f, S2_kernel %f\n", total_p, total_r,  memcpy_time, total_s1, total_s2);
	
	// printf("\n");

	time8 = get_time();

	t_start = high_resolution_clock::now();

	cudaFree(c);
	cudaFree(iN);
	cudaFree(iS);
	cudaFree(jE);
	cudaFree(jW);
	cudaFree(dN);
	cudaFree(dS);
	cudaFree(dE);
	cudaFree(dW);
	cudaFree(sums);
	cudaFree(sums2);

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Memory deallocation time: " << duration.count() << std::endl;

	mem_list.erase(mem_list.begin()+1);
	mem_list.erase(mem_list.begin()+1);
	mem_list.erase(mem_list.begin()+1);
	mem_list.erase(mem_list.begin()+1);
	mem_list.erase(mem_list.begin()+1);
	mem_list.erase(mem_list.begin()+1);
	mem_list.erase(mem_list.begin()+1);
	mem_list.erase(mem_list.begin()+1);
	mem_list.erase(mem_list.begin()+1);
	mem_list.erase(mem_list.begin()+1);
	mem_list.erase(mem_list.begin()+1);

	if(full == 1){
		ef_mem = page_align(page_align(mem_size_i) + page_align(mem_size) * 7);
		pre_bemps_free(tid, ef_mem);
		ef_cnt = 1;
	}
	
	//================================================================================80
	// 	SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
	//================================================================================80

	// cudaFree(c);
	// cudaFree(iN);
	// cudaFree(iS);
	// cudaFree(jE);
	// cudaFree(jW);
	// cudaFree(dN);
	// cudaFree(dS);
	// cudaFree(dE);
	// cudaFree(dW);
	// cudaFree(sums);
	// cudaFree(sums2);

	// int64_t eager_free_memory = (int64_t) 7 * (int64_t) mem_size + (int64_t) 2 * (int64_t) mem_size_i + (int64_t) 2 * (int64_t) mem_size_j;
	
	// if(full == 1){
	// 	pre_bemps_free(tid, eager_free_memory);
	// 	ef_cnt = 1;
	// }

	// t_start = high_resolution_clock::now();

	cudaEvent_t c_start, c_stop;
	CUDA_CHECK(cudaEventCreate(&c_start));
	CUDA_CHECK(cudaEventCreate(&c_stop));

	float total_c;

	cudaStream_t s6;
	CUDA_CHECK(cudaStreamCreateWithPriority(&s6, 0, priority));

	cudaEvent_t event6;
	CUDA_CHECK(cudaEventCreateWithFlags(&event6, cudaEventDisableTiming));

	CUDA_CHECK(cudaEventRecord(c_start));

	compress<<<blocks, threads, 0, s6>>>(	Ne,
									image);

	// compress<<<blocks, threads>>>(	Ne,
	// 	image);

	// if(full == 1){
	// 	nl_signal(tid);
	// }

	CUDA_CHECK(cudaEventRecord(event6, s6));

	task_monitoring(event6, tid, orig_alloc_mem, membytes);
								
	// CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaStreamDestroy(s6));
	CUDA_CHECK(cudaEventDestroy(event6));

	CUDA_CHECK(cudaEventRecord(c_stop));
	CUDA_CHECK(cudaEventSynchronize(c_stop));
	CUDA_CHECK(cudaEventElapsedTime(&total_c, c_start, c_stop));

	printf("C_kernel: %f\n", total_c);
	// t_stop = high_resolution_clock::now();
	// duration = duration_cast<milliseconds>(t_stop - t_start);
	// std::cout << "Tid: " << tid << " Kernel execution time: " << duration.count() << std::endl;

	// checkCUDAError("compress");

	time9 = get_time();

	//================================================================================80
	// 	COPY RESULTS BACK TO CPU
	//================================================================================80

	t_start = high_resolution_clock::now();

	// cudaMemcpy(tmp_image, image, mem_size, cudaMemcpyDeviceToHost);
	CUDA_CHECK(cudaMemPrefetchAsync(image, mem_size, cudaCpuDeviceId, 0));

	checkCUDAError("copy back");

	// CUDA_CHECK(cudaDeviceSynchronize());

	// memcpy(tmp_image, image, mem_size);

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Device to Host memcpy time: " << duration.count() << std::endl;

	time10 = get_time();

	t_start = high_resolution_clock::now();

	char* str1 = argv[5];
	char* str2 = "_nongds_output.txt";
	strcat(str1, str2);

	int o_fd = -1;
	o_fd = open(str1, O_CREAT | O_RDWR, 0664);
	if (o_fd < 0) {
		std::cerr << "file open error:" << std::endl;
	}

	printf("%d\n", o_fd);

	ssize_t o_ret = -1;
	o_ret = write(o_fd, image, mem_size);
	
	printf("%zd\n", o_ret);

	close(o_fd);

	// write_graphics(	"image_out.pgm",
	// 				tmp_image,
	// 				Nr,
	// 				Nc,
	// 				1,
	// 				255);

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Host to Storage memcpy time: " << duration.count() << std::endl;

	t_start = high_resolution_clock::now();

	// cudaFree(image_ori);
	cudaFree(image);

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
		
	//================================================================================80
	// 	WRITE IMAGE AFTER PROCESSING
	//================================================================================80

	time11 = get_time();

	//================================================================================80
	//	DEALLOCATE
	//================================================================================80

	// free(image_ori);
	// free(image);
	// free(iN); 
	// free(iS); 
	// free(jW); 
	// free(jE);

	time12 = get_time();

	//================================================================================80
	//	DISPLAY TIMING
	//================================================================================80

	printf("Time spent in different stages of the application:\n");
	printf("%15.12f s, %15.12f % : SETUP VARIABLES\n", 														(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : READ COMMAND LINE PARAMETERS\n", 										(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : READ IMAGE FROM FILE\n", 												(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU DRIVER INIT, CPU/GPU SETUP, MEMORY ALLOCATION\n", 														(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : RESIZE IMAGE\n", 					(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : COPY DATA TO CPU->GPU\n", 												(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : EXTRACT IMAGE\n", 														(float) (time7-time6) / 1000000, (float) (time7-time6) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : COMPUTE\n", 																(float) (time8-time7) / 1000000, (float) (time8-time7) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : COMPRESS IMAGE\n", 														(float) (time9-time8) / 1000000, (float) (time9-time8) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : COPY DATA TO GPU->CPU\n", 												(float) (time10-time9) / 1000000, (float) (time10-time9) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : SAVE IMAGE INTO FILE\n", 												(float) (time11-time10) / 1000000, (float) (time11-time10) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : FREE MEMORY\n", 															(float) (time12-time11) / 1000000, (float) (time12-time11) / (float) (time12-time0) * 100);
	printf("Total time:\n");
	printf("%.12f s\n", 																					(float) (time12-time0) / 1000000);

	clock_gettime( CLOCK_REALTIME, &specific_time);
	now = localtime(&specific_time.tv_sec);
	millsec = specific_time.tv_nsec;
  
	millsec = floor (specific_time.tv_nsec/1.0e6);
  
	printf("TID: %d Application end, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
		now->tm_mon + 1, now->tm_mday, now->tm_hour, 
		now->tm_min, now->tm_sec, millsec);
		
}

//====================================================================================================100
//	END OF FILE
//====================================================================================================100
