//========================================================================================================================================================================================================200
//======================================================================================================================================================150
//====================================================================================================100
//==================================================50

//========================================================================================================================================================================================================200
//	UPDATE
//========================================================================================================================================================================================================200

//	14 APR 2011 Lukasz G. Szafaryn

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdio.h>					// (in path known to compiler)			needed by printf
#include <stdlib.h>					// (in path known to compiler)			needed by malloc
#include <stdbool.h>				// (in path known to compiler)			needed by true/false

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./util/timer/timer.h"			// (in path specified here)
#include "./util/num/num.h"				// (in path specified here)

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./main.h"						// (in the current directory)

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel/kernel_gpu_cuda_wrapper.h"	// (in library path specified here)

#include "./kernel/kernel_gpu_cuda.cu"						// (in the current directory)	GPU kernel, cannot include with header file because of complications with passing of constant memory variables

#include <bemps.hpp>
#include <algorithm>
#include <vector>
#include <chrono>
#include <iostream>
#include <string.h>

#include "cufile.h"
#include <unistd.h>
#include <error.h>
#include <fcntl.h>
 #include <errno.h>
using namespace std::chrono;

#define CUDA_CHECK(val) { \
	if (val != cudaSuccess) { \
		fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(val), __LINE__, __FILE__); \
		exit(val); \
	} \
}

size_t page_align (size_t mem){
	if((mem % 2097152) != 0){
		size_t tmp = (2097152 * (mem / 2097152 + 1));
		return tmp;
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

//========================================================================================================================================================================================================200
//	MAIN FUNCTION
//========================================================================================================================================================================================================200

int 
main(	int argc, 
		char *argv [])
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

	int ef_cnt = 0;
	int ret_dev_id;
	size_t ef_mem;

	Parameters ret1;
	Parameters ret2;
	Parameters ret3;
	Parameters ret4;

	printf("thread block size of kernel = %d \n", NUMBER_THREADS);
	//======================================================================================================================================================150
	//	CPU/MCPU VARIABLES
	//======================================================================================================================================================150

	// timer
	long long time0;

	time0 = get_time();

	// timer
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long time6;
	long long time7;

	// counters
	int i, j, k, l, m, n;

	// system memory
	par_str par_cpu;
	dim_str dim_cpu;
	box_str* box_cpu;
	FOUR_VECTOR* rv_cpu;
	fp* qv_cpu;
	FOUR_VECTOR* fv_cpu;

	box_str* tmp_box_cpu;
	FOUR_VECTOR* tmp_rv_cpu;
	fp* tmp_qv_cpu;
	FOUR_VECTOR* tmp_fv_cpu;

	int nh;

	time1 = get_time();

	//======================================================================================================================================================150
	//	CHECK INPUT ARGUMENTS
	//======================================================================================================================================================150

	// assing default values
	dim_cpu.boxes1d_arg = 1;

	// go through arguments
	for(dim_cpu.cur_arg=1; dim_cpu.cur_arg<argc; dim_cpu.cur_arg++){
		// check if -boxes1d
		if(strcmp(argv[dim_cpu.cur_arg], "-boxes1d")==0){
			// check if value provided
			if(argc>=dim_cpu.cur_arg+1){
				// check if value is a number
				if(isInteger(argv[dim_cpu.cur_arg+1])==1){
					dim_cpu.boxes1d_arg = atoi(argv[dim_cpu.cur_arg+1]);
					if(dim_cpu.boxes1d_arg<0){
						printf("ERROR: Wrong value to -boxes1d parameter, cannot be <=0\n");
						return 0;
					}
					// tid = atoi(argv[3]);
					dim_cpu.cur_arg = dim_cpu.cur_arg+1;
					break;
				}
				// value is not a number
				else{
					printf("ERROR: Value to -boxes1d parameter in not a number\n");
					return 0;
				}
			}
			// value not provided
			else{
				printf("ERROR: Missing value to -boxes1d parameter\n");
				return 0;
			}
		}
		// unknown
		else{
			printf("ERROR: Unknown parameter\n");
			return 0;
		}
	}

	// Print configuration
	printf("Configuration used: boxes1d = %d\n", dim_cpu.boxes1d_arg);

	time2 = get_time();

	//======================================================================================================================================================150
	//	INPUTS
	//======================================================================================================================================================150

	par_cpu.alpha = 0.5;

	time3 = get_time();

	//======================================================================================================================================================150
	//	DIMENSIONS
	//======================================================================================================================================================150

	// total number of boxes
	dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;

	// how many particles space has in each direction
	dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
	dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
	dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

	// box array
	dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

	time4 = get_time();

	//======================================================================================================================================================150
	//	SYSTEM MEMORY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	BOX
	//====================================================================================================100

	dim3 threads;
	dim3 blocks;

	blocks.x = dim_cpu.number_boxes;
	blocks.y = 1;
	threads.x = NUMBER_THREADS;											// define the number of threads in the block
	threads.y = 1;

	// tmp_box_cpu = (box_str*)malloc(dim_cpu.box_mem);
	// tmp_rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
	// tmp_qv_cpu = (fp*)malloc(dim_cpu.space_mem2);
	// tmp_fv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);

	// printf("HI1\n");

	// initialize number of home boxes
	nh = 0;

	// for(i=0; i<dim_cpu.boxes1d_arg; i++){
	// 	// home boxes in y direction
	// 	for(j=0; j<dim_cpu.boxes1d_arg; j++){
	// 		// home boxes in x direction
	// 		for(k=0; k<dim_cpu.boxes1d_arg; k++){

	// 			// current home box
	// 			tmp_box_cpu[nh].x = k;
	// 			tmp_box_cpu[nh].y = j;
	// 			tmp_box_cpu[nh].z = i;
	// 			tmp_box_cpu[nh].number = nh;
	// 			tmp_box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

	// 			// initialize number of neighbor boxes
	// 			tmp_box_cpu[nh].nn = 0;

	// 			// neighbor boxes in z direction
	// 			for(l=-1; l<2; l++){
	// 				// neighbor boxes in y direction
	// 				for(m=-1; m<2; m++){
	// 					// neighbor boxes in x direction
	// 					for(n=-1; n<2; n++){

	// 						// check if (this neighbor exists) and (it is not the same as home box)
	// 						if(		(((i+l)>=0 && (j+m)>=0 && (k+n)>=0)==true && ((i+l)<dim_cpu.boxes1d_arg && (j+m)<dim_cpu.boxes1d_arg && (k+n)<dim_cpu.boxes1d_arg)==true)	&&
	// 								(l==0 && m==0 && n==0)==false	){

	// 							// current neighbor box
	// 							tmp_box_cpu[nh].nei[tmp_box_cpu[nh].nn].x = (k+n);
	// 							tmp_box_cpu[nh].nei[tmp_box_cpu[nh].nn].y = (j+m);
	// 							tmp_box_cpu[nh].nei[tmp_box_cpu[nh].nn].z = (i+l);
	// 							tmp_box_cpu[nh].nei[tmp_box_cpu[nh].nn].number =	(tmp_box_cpu[nh].nei[tmp_box_cpu[nh].nn].z * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) + 
	// 																		(tmp_box_cpu[nh].nei[tmp_box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) + 
	// 																		tmp_box_cpu[nh].nei[tmp_box_cpu[nh].nn].x;
	// 							tmp_box_cpu[nh].nei[tmp_box_cpu[nh].nn].offset = tmp_box_cpu[nh].nei[tmp_box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

	// 							// increment neighbor box
	// 							tmp_box_cpu[nh].nn = tmp_box_cpu[nh].nn + 1;

	// 						}

	// 					} // neighbor boxes in x direction
	// 				} // neighbor boxes in y direction
	// 			} // neighbor boxes in z direction

	// 			// increment home box
	// 			nh = nh + 1;

	// 		} // home boxes in x direction
	// 	} // home boxes in y direction
	// } // home boxes in z direction

	//====================================================================================================100
	//	PARAMETERS, DISTANCE, CHARGE AND FORCE
	//====================================================================================================100

	// random generator seed set to random value - time in this case
	srand(time(NULL));

	// input (distances)
	// change to UVM
	// rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
	
	// printf("HI2\n");

	// for(i=0; i<dim_cpu.space_elem; i=i+1){
	// 	tmp_rv_cpu[i].v = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
	// 	tmp_rv_cpu[i].x = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
	// 	tmp_rv_cpu[i].y = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
	// 	tmp_rv_cpu[i].z = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
	// }

	// input (charge)
	// change to UVM
	// qv_cpu = (fp*)malloc(dim_cpu.space_mem2);

	// for(i=0; i<dim_cpu.space_elem; i=i+1){
	// 	tmp_qv_cpu[i] = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
	// }

	// printf("HI3\n");

	// output (forces)
	// change to UVM
	// fv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);

	// for(i=0; i<dim_cpu.space_elem; i=i+1){
	// 	tmp_fv_cpu[i].v = 0;								// set to 0, because kernels keeps adding to initial value
	// 	tmp_fv_cpu[i].x = 0;								// set to 0, because kernels keeps adding to initial value
	// 	tmp_fv_cpu[i].y = 0;								// set to 0, because kernels keeps adding to initial value
	// 	tmp_fv_cpu[i].z = 0;								// set to 0, because kernels keeps adding to initial value
	// }

	// printf("HI4\n");

	// printf("blocks_x: %d, blocks_y: %d, threads_x: %d, threads_y: %d\n", blocks.x, blocks.y, threads.x, threads.y);

	size_t membytes = 0;
	membytes += page_align(dim_cpu.box_mem);
	membytes += page_align(dim_cpu.space_mem);
	membytes += page_align(dim_cpu.space_mem2);
	membytes += page_align(dim_cpu.space_mem);
	membytes += 309 * 1024 * 1024;

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

	// full = 0;
	// int* dummy;
	// size_t d_size = 7000000000;
	// CUDA_CHECK(cudaMalloc(&dummy, d_size));
	// CUDA_CHECK(cudaMemset(dummy, 0, d_size));

	printf("Full: %d\n", full);

  	printf("ret_dev_id: %d\n", ret_dev_id);

	// allocate boxes
	// change to UVM
	// box_cpu = (box_str*)malloc(dim_cpu.box_mem); 
	
	auto t_start = high_resolution_clock::now();

	CUDA_CHECK(cudaMallocManaged(&box_cpu, dim_cpu.box_mem));
	CUDA_CHECK(cudaMallocManaged(&rv_cpu, dim_cpu.space_mem));
	CUDA_CHECK(cudaMallocManaged(&qv_cpu, dim_cpu.space_mem2));
	CUDA_CHECK(cudaMallocManaged(&fv_cpu, dim_cpu.space_mem));

	auto t_stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Device mem allocation time: " << duration.count() << std::endl;

	ret1.devPtr = box_cpu;
	ret1.advice = cudaMemAdviseSetPreferredLocation;
	ret1.device = ret_dev_id;
	ret1.alloc_size = dim_cpu.box_mem;

  	ret2.devPtr = rv_cpu;
	ret2.advice = cudaMemAdviseSetPreferredLocation;
	ret2.device = ret_dev_id;
	ret2.alloc_size = dim_cpu.space_mem;

  	ret3.devPtr = qv_cpu;
	ret3.advice = cudaMemAdviseSetPreferredLocation;
	ret3.device = ret_dev_id;
	ret3.alloc_size = dim_cpu.space_mem2;

  	ret4.devPtr = fv_cpu;
	ret4.advice = cudaMemAdviseSetPreferredLocation;
	ret4.device = ret_dev_id;
	ret4.alloc_size = dim_cpu.space_mem;

	mem_list.push_back(ret1);
	mem_list.push_back(ret2);
	mem_list.push_back(ret3);
	mem_list.push_back(ret4);

	if(full == 1){
		for(Parameters var : mem_list){
			CUDA_CHECK(cudaMemAdvise(var.devPtr, var.alloc_size, var.advice, var.device));
			// CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, ret.device, 0));
		}
		// CUDA_CHECK(cudaMemcpy(box_cpu, tmp_box_cpu, dim_cpu.box_mem, cudaMemcpyHostToDevice));
		// CUDA_CHECK(cudaMemcpy(rv_cpu, tmp_rv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice));
		// CUDA_CHECK(cudaMemcpy(qv_cpu, tmp_qv_cpu, dim_cpu.space_mem2, cudaMemcpyHostToDevice));
		// CUDA_CHECK(cudaMemcpy(fv_cpu, tmp_fv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice));
	}

	// initialize number of home boxes
	nh = 0;

	t_start = high_resolution_clock::now();

	// memcpy(box_cpu, tmp_box_cpu, dim_cpu.box_mem);
	// memcpy(rv_cpu, tmp_rv_cpu, dim_cpu.space_mem);
	// memcpy(qv_cpu, tmp_qv_cpu, dim_cpu.space_mem2);
	// memcpy(fv_cpu, tmp_fv_cpu, dim_cpu.space_mem);

	CUfileError_t status;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
	int fd = -1;
	ssize_t ret = -1;

	status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
            std::cerr << "cufile driver open error: " << std::endl;
         // << cuFileGetErrorString(status) << std::endl;
            return -1;
    }

	if(dim_cpu.boxes1d_arg == 100){
		printf("Large1\n");
		fd = open("large1/box_large1.txt", O_RDONLY | O_DIRECT, 0644);
		memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		ret = -1;
 
		ret = cuFileRead(cf_handle, box_cpu, dim_cpu.box_mem, 0, 0);
		cuFileHandleDeregister(cf_handle);
     	close(fd);

		fd = open("large1/rv_large1.txt", O_RDONLY | O_DIRECT, 0644);
		memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		ret = -1;
  
		ret = cuFileRead(cf_handle, rv_cpu, dim_cpu.space_mem, 0, 0);
		cuFileHandleDeregister(cf_handle);
		close(fd);

		fd = open("large1/qv_large1.txt", O_RDONLY | O_DIRECT, 0644);
		memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		ret = -1;
  
		ret = cuFileRead(cf_handle, qv_cpu, dim_cpu.space_mem2, 0, 0);
		cuFileHandleDeregister(cf_handle);
		close(fd);

		fd = open("large1/fv_large1.txt", O_RDONLY | O_DIRECT, 0644);
		memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		ret = -1;
  
		ret = cuFileRead(cf_handle, fv_cpu, dim_cpu.space_mem, 0, 0);
		cuFileHandleDeregister(cf_handle);
		close(fd);
	}
	else if(dim_cpu.boxes1d_arg == 110){
		printf("Large2\n");
		fd = open("large2/box_large2.txt", O_RDONLY | O_DIRECT, 0644);
		memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		ret = -1;
 
		ret = cuFileRead(cf_handle, box_cpu, dim_cpu.box_mem, 0, 0);
		cuFileHandleDeregister(cf_handle);
     	close(fd);

		fd = open("large2/rv_large2.txt", O_RDONLY | O_DIRECT, 0644);
		memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		ret = -1;
  
		ret = cuFileRead(cf_handle, rv_cpu, dim_cpu.space_mem, 0, 0);
		cuFileHandleDeregister(cf_handle);
		close(fd);

		fd = open("large2/qv_large2.txt", O_RDONLY | O_DIRECT, 0644);
		memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		ret = -1;
  
		ret = cuFileRead(cf_handle, qv_cpu, dim_cpu.space_mem2, 0, 0);
		cuFileHandleDeregister(cf_handle);
		close(fd);

		fd = open("large2/fv_large2.txt", O_RDONLY | O_DIRECT, 0644);
		memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		ret = -1;
  
		ret = cuFileRead(cf_handle, fv_cpu, dim_cpu.space_mem, 0, 0);
		cuFileHandleDeregister(cf_handle);
		close(fd);
	}
	else if(dim_cpu.boxes1d_arg == 120){
		printf("Large3\n");
		fd = open("large3/box_large3.txt", O_RDONLY | O_DIRECT, 0644);
		memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		ret = -1;
 
		ret = cuFileRead(cf_handle, box_cpu, dim_cpu.box_mem, 0, 0);
		cuFileHandleDeregister(cf_handle);
     	close(fd);

		fd = open("large3/rv_large3.txt", O_RDONLY | O_DIRECT, 0644);
		memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		ret = -1;
  
		ret = cuFileRead(cf_handle, rv_cpu, dim_cpu.space_mem, 0, 0);
		cuFileHandleDeregister(cf_handle);
		close(fd);

		fd = open("large3/qv_large3.txt", O_RDONLY | O_DIRECT, 0644);
		memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		ret = -1;
  
		ret = cuFileRead(cf_handle, qv_cpu, dim_cpu.space_mem2, 0, 0);
		cuFileHandleDeregister(cf_handle);
		close(fd);

		fd = open("large3/fv_large3.txt", O_RDONLY | O_DIRECT, 0644);
		memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
		cf_descr.handle.fd = fd;
		cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
		status = cuFileHandleRegister(&cf_handle, &cf_descr);
		ret = -1;
  
		ret = cuFileRead(cf_handle, fv_cpu, dim_cpu.space_mem, 0, 0);
		cuFileHandleDeregister(cf_handle);
		close(fd);
	}

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Storage to Device memcpy time: " << duration.count() << std::endl;

	time5 = get_time();

	//======================================================================================================================================================150
	//	KERNEL
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU_CUDA
	//====================================================================================================100

	size_t free_mem, total_mem;
	// CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

	// printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

	t_start = high_resolution_clock::now();

	// if(full == 1){
	// 	for(Parameters ret : mem_list){
	// 		// CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, ret.device, 0));
	// 		CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
	// 	}
	// }

	// CUDA_CHECK(cudaDeviceSynchronize());

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Host to Device memcpy time: " << duration.count() << std::endl;

	cudaEvent_t k_start, k_stop;
	CUDA_CHECK(cudaEventCreate(&k_start));
	CUDA_CHECK(cudaEventCreate(&k_stop));

	float total_k;

	cudaStream_t s1;
	CUDA_CHECK(cudaStreamCreateWithPriority(&s1, 0, priority));

	cudaEvent_t event1;
	CUDA_CHECK(cudaEventCreateWithFlags(&event1, cudaEventDisableTiming));

	CUDA_CHECK(cudaEventRecord(k_start));

	// launch kernel - all boxes
	kernel_gpu_cuda<<<blocks, threads, 0, s1>>>(	par_cpu,
											dim_cpu,
											box_cpu,
											rv_cpu,
											qv_cpu,
											fv_cpu);

	CUDA_CHECK(cudaEventRecord(event1, s1));

	task_monitoring(event1, tid, orig_alloc_mem, membytes);
											  
	CUDA_CHECK(cudaEventRecord(k_stop));
	CUDA_CHECK(cudaEventSynchronize(k_stop));
	CUDA_CHECK(cudaEventElapsedTime(&total_k, k_start, k_stop));
											  
	printf("First_kernel: %f\n", total_k);

	// kernel_gpu_cuda_wrapper(par_cpu,
	// 						dim_cpu,
	// 						box_cpu,
	// 						rv_cpu,
	// 						qv_cpu,
	// 						fv_cpu,
	// 						os_perc);

	// CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

	// printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

	CUDA_CHECK(cudaFree(box_cpu));
	CUDA_CHECK(cudaFree(rv_cpu));
	CUDA_CHECK(cudaFree(qv_cpu));

	// CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

	// printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

	if(full == 1){
		ef_mem = page_align(dim_cpu.box_mem) + page_align(dim_cpu.space_mem) + page_align(dim_cpu.space_mem2);
		pre_bemps_free(tid, ef_mem);
		ef_cnt = 1;
	}

	time6 = get_time();

	//======================================================================================================================================================150
	//	SYSTEM MEMORY DEALLOCATION
	//======================================================================================================================================================150

	// dump results
#ifdef OUTPUT
	CUDA_CHECK(cudaMemPrefetchAsync(ret4.devPtr, ret4.alloc_size, cudaCpuDeviceId, 0))
	// CUDA_CHECK(cudaDeviceSynchronize());
	printf("Output!!!!\n");
    FILE *fptr;
	fptr = fopen("result.txt", "w");	
	for(i=0; i<dim_cpu.space_elem; i=i+1){
        	fprintf(fptr, "%f, %f, %f, %f\n", fv_cpu[i].v, fv_cpu[i].x, fv_cpu[i].y, fv_cpu[i].z);
	}
	fclose(fptr);
#endif       	

	t_start = high_resolution_clock::now();

	char* str1 = argv[3];
	char* str2 = "_gds_output.txt";
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

	ret = cuFileWrite(cf_handle, fv_cpu, dim_cpu.space_mem, 0, 0);
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

	CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

	printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

	t_start = high_resolution_clock::now();

	CUDA_CHECK(cudaFree(fv_cpu));

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

	time7 = get_time();

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

	// printf("Time spent in different stages of the application:\n");

	// printf("%15.12f s, %15.12f % : VARIABLES\n",						(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time7-time0) * 100);
	// printf("%15.12f s, %15.12f % : INPUT ARGUMENTS\n", 					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time7-time0) * 100);
	// printf("%15.12f s, %15.12f % : INPUTS\n",							(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time7-time0) * 100);
	// printf("%15.12f s, %15.12f % : dim_cpu\n", 							(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time7-time0) * 100);
	// printf("%15.12f s, %15.12f % : SYS MEM: ALO\n",						(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time7-time0) * 100);

	// printf("%15.12f s, %15.12f % : KERNEL: COMPUTE\n",					(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time7-time0) * 100);

	// printf("%15.12f s, %15.12f % : SYS MEM: FRE\n", 					(float) (time7-time6) / 1000000, (float) (time7-time6) / (float) (time7-time0) * 100);

	// printf("Total time:\n");
	// printf("%.12f s\n", 												(float) (time7-time0) / 1000000);

	//======================================================================================================================================================150
	//	RETURN
	//======================================================================================================================================================150
	clock_gettime( CLOCK_REALTIME, &specific_time);
	now = localtime(&specific_time.tv_sec);
	millsec = specific_time.tv_nsec;
  
	millsec = floor (specific_time.tv_nsec/1.0e6);
  
	printf("TID: %d Application end, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
		now->tm_mon + 1, now->tm_mday, now->tm_hour, 
		now->tm_min, now->tm_sec, millsec);
		
	return 0.0;																					// always returns 0.0

}
