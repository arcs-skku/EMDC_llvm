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

#include "device.c"				// (in library path specified to compiler)	needed by for device functions

#include <chrono>
#include <iostream>

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
		return (2097152 * (mem / 2097152));
	}
	else{
		return mem;
	}
}

bool full; // 1 = fully secured

struct Parameters{
	void* devPtr;
	size_t count;
	cudaMemoryAdvise advice;
	int device;
	size_t alloc_size; // 디바이스 메모리에 올릴 페이지 크기
	// std::bitset<kernel_num> bit; // liveness check
};

void init (fp* arr, int size){
	for(int i = 0; i < size / sizeof(fp); i++){
		arr[i] = 0;
	}
}

// void task_monitoring(cudaEvent_t event, int tid, long orig_alloc_mem, size_t membytes){
// 	long update_mem = 0;
// 	long tmp_mem = 0;
// 	update_mem = bemps_extra_task_mem(tid);
// 	tmp_mem = update_mem;
// 	if(full != 1){
// 		while(1){
// 			bool chk_former_task = 0;
// 			update_mem = bemps_extra_task_mem(tid);
// 			cudaStream_t s_e;
// 			CUDA_CHECK(cudaStreamCreate(&s_e));
// 			if(orig_alloc_mem != update_mem){
// 				chk_former_task = 1;
// 			}
// 			if(cudaEventQuery(event) == cudaSuccess){
// 				printf("Kernel End\n");
// 				break;
// 			}
// 			if((chk_former_task == 1) && (full != 1)){
// 				if(update_mem == membytes){
// 					full = 1;
// 				}
// 				if(full == 1){
// 					printf("Hello\n");
// 					for(Parameters ret : mem_list){
// 						CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
// 						CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, ret.device, 0));
// 					}
// 					break;
// 				}
// 			}
// 			CUDA_CHECK(cudaStreamDestroy(s_e));
// 		}
// 	}
// }

//====================================================================================================100
//	MAIN FUNCTION
//====================================================================================================100

int main(int argc, char *argv []){

	//================================================================================80
	// 	VARIABLES
	//================================================================================80
	size_t k1_mem_req = 901775360;
  	size_t k2_mem_req = 2705326080;
	size_t k3_mem_req = 2705326080;
  	size_t k4_mem_req = 7216300032;
	size_t k5_mem_req = 7216300032;
  	size_t k6_mem_req = 7216300032;

	// size_t k1_mem_req = 1600126976;
  	// size_t k2_mem_req = 4800380928;
	// size_t k3_mem_req = 4800380928;
  	// size_t k4_mem_req = 12803112960;
	// size_t k5_mem_req = 12803112960;
  	// size_t k6_mem_req = 12803112960;

	int ef_cnt = 0;

	int ret_dev_id;
	int os_perc = atoi(argv[5]);
	
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
	// int* iN;
	// int* iS;
	// int* jE;
	// int* jW;
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

	threads.x = NUMBER_THREADS;												// define the number of threads in the block
	threads.y = 1;
	blocks_x = Ne/threads.x;
	if (Ne % threads.x != 0){												// compensate for division remainder above by adding one grid
		blocks_x = blocks_x + 1;																	
	}
	blocks.x = blocks_x;													// define the number of blocks in the grid
	blocks.y = 1;

	fp* zzam;
	zzam = (fp*)malloc(mem_size);

	printf("OS_perc: %d\n", os_perc);

	size_t f_free_mem, f_total_mem;
	size_t free_mem, total_mem;
	CUDA_CHECK(cudaMemGetInfo(&f_free_mem, &f_total_mem));

	printf("Free: %zd, Total: %zd\n", f_free_mem, f_total_mem);

	float dummy_perc = ((float)os_perc / (float)100);
	printf("Dummy percentage: %f\n", dummy_perc);

	int* dummy;
	size_t dummy_size;

	struct timespec specific_time;
    struct tm *now;
    int millsec;
    clock_gettime( CLOCK_REALTIME, &specific_time);
    now = localtime(&specific_time.tv_sec);
    millsec = specific_time.tv_nsec;

    millsec = floor (specific_time.tv_nsec/1.0e6);

    printf("Before strat, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", 1900 + now->tm_year, 
        now->tm_mon + 1, now->tm_mday, now->tm_hour, 
        now->tm_min, now->tm_sec, millsec);

	cudaMallocManaged(&image_ori, sizeof(fp) * image_ori_elem);

	read_graphics(	"../../../data/srad/image.pgm",
								image_ori,
								image_ori_rows,
								image_ori_cols,
								1);

	time3 = get_time();

	//================================================================================80
	// 	KERNEL EXECUTION PARAMETERS
	//================================================================================80

	time4 = get_time();

	//================================================================================80
	// 	RESIZE IMAGE (ASSUMING COLUMN MAJOR STORAGE OF image_orig)
	//================================================================================80

	cudaMallocManaged(&image, sizeof(fp) * Ne);

	resize(	image_ori,
				image_ori_rows,
				image_ori_cols,
				image,
				Nr,
				Nc,
				1);

	time5 = get_time();

	//================================================================================80
	// 	SETUP
	//================================================================================80

	cudaMallocManaged(&iN, mem_size_i);
	cudaMallocManaged(&iS, mem_size_i);

	cudaMallocManaged(&jW, mem_size_j);
	cudaMallocManaged(&jE, mem_size_j);

	// N/S/W/E indices of surrounding pixels (every element of IMAGE)
	for (i=0; i<Nr; i++) {
		iN[i] = i-1;														// holds index of IMAGE row above
		iS[i] = i+1;														// holds index of IMAGE row below
	}
	for (j=0; j<Nc; j++) {
		jW[j] = j-1;														// holds index of IMAGE column on the left
		jE[j] = j+1;														// holds index of IMAGE column on the right
	}

	// N/S/W/E boundary conditions, fix surrounding indices outside boundary of image
	iN[0]    = 0;															// changes IMAGE top row index from -1 to 0
	iS[Nr-1] = Nr-1;														// changes IMAGE bottom row index from Nr to Nr-1 
	jW[0]    = 0;															// changes IMAGE leftmost column index from -1 to 0
	jE[Nc-1] = Nc-1;														// changes IMAGE rightmost column index from Nc to Nc-1

	cudaMallocManaged(&sums, mem_size);
	cudaMallocManaged(&sums2, mem_size);
	cudaMallocManaged(&dN, mem_size);
	cudaMallocManaged(&dS, mem_size);
	cudaMallocManaged(&dW, mem_size);
	cudaMallocManaged(&dE, mem_size);
	cudaMallocManaged(&c, mem_size);

	// init(sums, mem_size);
	// init(sums2, mem_size);
	// init(dN, mem_size);
	// init(dS, mem_size);
	// init(dW, mem_size);
	// init(dE, mem_size);
	// init(c, mem_size);

	//================================================================================80
	// 	COPY INPUT TO CPU
	//================================================================================80

	time6 = get_time();

	//================================================================================80
	// 	SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
	//================================================================================80

	// CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

	// printf("Before e Kernel Free: %zd, Total: %zd\n", free_mem, total_mem);

	dummy_size = page_align(f_free_mem * dummy_perc) - page_align(k1_mem_req);
	printf("Dummy size: %zd\n", dummy_size);
	
	// CUDA_CHECK(cudaMalloc((void**) &dummy, dummy_size))
	// CUDA_CHECK(cudaMemset(dummy, 0, dummy_size));

	cudaEvent_t e_start, e_stop;
	CUDA_CHECK(cudaEventCreate(&e_start));
	CUDA_CHECK(cudaEventCreate(&e_stop));

	float total_e, time;

	CUDA_CHECK(cudaEventRecord(e_start));

	extract<<<blocks, threads>>>(	Ne,
		image);

	CUDA_CHECK(cudaEventRecord(e_stop));
	CUDA_CHECK(cudaEventSynchronize(e_stop));
	CUDA_CHECK(cudaEventElapsedTime(&total_e, e_start, e_stop));

	printf("E_kernel: %f\n", total_e);

	// cudaFree(dummy);

	// CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

	// printf("After e Kernel Free: %zd, Total: %zd\n", free_mem, total_mem);

	// CUDA_CHECK(cudaMemcpy(zzam, image, mem_size, cudaMemcpyDeviceToHost));
	// // CUDA_CHECK(cudaDeviceSynchronize());

	// CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

	// printf("After Prefetch Free: %zd, Total: %zd\n", free_mem, total_mem);

	time7 = get_time();

	//================================================================================80
	// 	COMPUTATION
	//================================================================================80

	cudaEvent_t *p_start, *p_stop, *r_start, *r_stop, *s1_start, *s1_stop, *s2_start, *s2_stop, *cm_start, *cm_stop;

	p_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * niter);
	p_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * niter);
	r_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * niter * 4);
	r_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * niter * 4);
	s1_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * niter);
	s1_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * niter);
	s2_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * niter);
	s2_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * niter);
	cm_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * niter);
	cm_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * niter);

	float total_p, total_r, total_s1, total_s2, memcpy_time;
	total_p = 0;
	total_r = 0;
	total_s1 = 0;
	total_s2 = 0;
	memcpy_time = 0;

	cudaEvent_t event2, event3;

	for (iter=0; iter<niter; iter++){										// do for the number of iterations input parameter
		int r_iter = 0;
	// printf("%d ", iter);
	// fflush(NULL);

		if(iter == 0){
			dummy_size = page_align(f_free_mem * dummy_perc) - page_align(k2_mem_req);
		}
		else{
			dummy_size = page_align(f_free_mem * dummy_perc) - page_align(k4_mem_req);
		}
		printf("K2 Dummy size: %zd\n", dummy_size);
		
		CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

		printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

		// CUDA_CHECK(cudaMalloc((void**) &dummy, dummy_size))
		// CUDA_CHECK(cudaMemset(dummy, 0, dummy_size));

		CUDA_CHECK(cudaEventCreate(&p_start[iter]));
		CUDA_CHECK(cudaEventCreate(&p_stop[iter]));
		CUDA_CHECK(cudaEventRecord(p_start[iter]));

		prepare<<<blocks, threads>>>(	Ne,
			image,
			sums,
			sums2);

		CUDA_CHECK(cudaEventRecord(p_stop[iter]));

		CUDA_CHECK(cudaDeviceSynchronize());
		// cudaFree(dummy);

		// performs subsequent reductions of sums
		blocks2.x = blocks.x;												// original number of blocks
		blocks2.y = blocks.y;												
		no = Ne;														// original number of sum elements
		mul = 1;														// original multiplier

		while(blocks2.x != 0){

			checkCUDAError("before reduce");

			if(iter == 0){
				dummy_size = page_align(f_free_mem * dummy_perc) - page_align(k3_mem_req);
			}
			else{
				dummy_size = page_align(f_free_mem * dummy_perc) - page_align(k4_mem_req);
			}
			printf("K3 Dummy size: %zd\n", dummy_size);
			
			// CUDA_CHECK(cudaMalloc((void**) &dummy, dummy_size))
			// CUDA_CHECK(cudaMemset(dummy, 0, dummy_size));

			CUDA_CHECK(cudaEventCreate(&r_start[iter*4+r_iter]));
			CUDA_CHECK(cudaEventCreate(&r_stop[iter*4+r_iter]));
			CUDA_CHECK(cudaEventRecord(r_start[iter*4+r_iter]));

			reduce<<<blocks2, threads>>>(	Ne,
				no,
				mul,
				sums, 
				sums2);
			
			CUDA_CHECK(cudaEventRecord(r_stop[iter*4+r_iter]));

			CUDA_CHECK(cudaDeviceSynchronize());
			// cudaFree(dummy);

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

		checkCUDAError("before copy sum");

		CUDA_CHECK(cudaEventCreate(&cm_start[iter]));
		CUDA_CHECK(cudaEventCreate(&cm_stop[iter]));
		CUDA_CHECK(cudaEventRecord(cm_start[iter]));

		// copy total sums to device
		mem_size_single = sizeof(fp) * 1;
		// cudaMemcpy(&total, d_sums, mem_size_single, cudaMemcpyDeviceToHost);
		total = sums[0];
		// cudaMemcpy(&total2, d_sums2, mem_size_single, cudaMemcpyDeviceToHost);
		total2 = sums2[0];

		checkCUDAError("copy sum");

		CUDA_CHECK(cudaEventRecord(cm_stop[iter]));

		// calculate statistics
		meanROI	= sums[0] / fp(NeROI);										// gets mean (average) value of element in ROI
		meanROI2 = meanROI * meanROI;										//
		varROI = (sums2[0] / fp(NeROI)) - meanROI2;						// gets variance of ROI								
		q0sqr = varROI / meanROI2;											// gets standard deviation of ROI

		dummy_size = page_align(f_free_mem * dummy_perc) - page_align(k4_mem_req);
		printf("K4 Dummy size: %zd\n", dummy_size);
		
		CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

		printf("Free: %zd, Total: %zd\n", f_free_mem, f_total_mem);

		// CUDA_CHECK(cudaMalloc((void**) &dummy, dummy_size))
		// CUDA_CHECK(cudaMemset(dummy, 0, dummy_size));

		CUDA_CHECK(cudaEventCreate(&s1_start[iter]));
		CUDA_CHECK(cudaEventCreate(&s1_stop[iter]));
		CUDA_CHECK(cudaEventRecord(s1_start[iter]));

		srad<<<blocks, threads>>>(	lambda,									// SRAD coefficient 
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

		CUDA_CHECK(cudaEventRecord(s1_stop[iter]));

		CUDA_CHECK(cudaDeviceSynchronize());
		// cudaFree(dummy);

		dummy_size = page_align(f_free_mem * dummy_perc) - page_align(k5_mem_req);
		printf("K5 Dummy size: %zd\n", dummy_size);
		
		CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

		printf("Free: %zd, Total: %zd\n", f_free_mem, f_total_mem);

		// CUDA_CHECK(cudaMalloc((void**) &dummy, dummy_size))
		// CUDA_CHECK(cudaMemset(dummy, 0, dummy_size));

		CUDA_CHECK(cudaEventCreate(&s2_start[iter]));
		CUDA_CHECK(cudaEventCreate(&s2_stop[iter]));
		CUDA_CHECK(cudaEventRecord(s2_start[iter]));

		srad2<<<blocks, threads>>>(	lambda,									// SRAD coefficient 
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

		CUDA_CHECK(cudaEventRecord(s2_stop[iter]));

		CUDA_CHECK(cudaDeviceSynchronize());
		// cudaFree(dummy);

		// checkCUDAError("srad2");

	}

	CUDA_CHECK(cudaEventSynchronize(p_stop[niter - 1]));
	CUDA_CHECK(cudaEventSynchronize(r_stop[niter * 4 - 1]));
	CUDA_CHECK(cudaEventSynchronize(s1_stop[niter - 1]));
	CUDA_CHECK(cudaEventSynchronize(s2_stop[niter - 1]));

	for(int i = 0; i < niter; i++){
		cudaEventElapsedTime(&time, p_start[i], p_stop[i]);
		printf("%f ", time);
		total_p += time;
		for(int j = 0; j < 4; j++){
			cudaEventElapsedTime(&time, r_start[i*4+j], r_stop[i*4+j]);
			printf("%f ", time);
			total_r += time;
		}
		// printf("%f ", time);
		cudaEventElapsedTime(&time, cm_start[i], cm_stop[i]);
		printf("%f ", time);
		memcpy_time += time;
		cudaEventElapsedTime(&time, s1_start[i], s1_stop[i]);
		printf("%f ", time);
		total_s1 += time;
		cudaEventElapsedTime(&time, s2_start[i], s2_stop[i]);
		printf("%f\n", time);
		total_s2 += time;
	}
	


	printf("P_kernel: %f, R_kernel: %f, Memcpy: %f, S1_kernel: %f, S2_kernel %f\n", total_p, total_r,  memcpy_time, total_s1, total_s2);

	// printf("\n");

	time8 = get_time();

	//================================================================================80
	// 	SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
	//================================================================================80

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

	dummy_size = page_align(f_free_mem * dummy_perc) - page_align(k6_mem_req);
	printf("K6 Dummy size: %zd\n", dummy_size);
	
	CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

	printf("Free: %zd, Total: %zd\n", f_free_mem, f_total_mem);

	// CUDA_CHECK(cudaMalloc((void**) &dummy, dummy_size))
	// CUDA_CHECK(cudaMemset(dummy, 0, dummy_size));

	cudaEvent_t c_start, c_stop;
	CUDA_CHECK(cudaEventCreate(&c_start));
	CUDA_CHECK(cudaEventCreate(&c_stop));

	float total_c;

	CUDA_CHECK(cudaEventRecord(c_start));

	compress<<<blocks, threads>>>(	Ne,
		image);

	CUDA_CHECK(cudaEventRecord(c_stop));
	CUDA_CHECK(cudaEventSynchronize(c_stop));
	CUDA_CHECK(cudaEventElapsedTime(&total_c, c_start, c_stop));

	printf("C_kernel: %f\n", total_c);

	time9 = get_time();

	//================================================================================80
	// 	COPY RESULTS BACK TO CPU
	//================================================================================80

	// cudaMemcpy(image, d_I, mem_size, cudaMemcpyDeviceToHost);
	cudaMemPrefetchAsync(image, mem_size, cudaCpuDeviceId, 0);

	checkCUDAError("copy back");

	CUDA_CHECK(cudaDeviceSynchronize());

	time10 = get_time();

	//================================================================================80
	// 	WRITE IMAGE AFTER PROCESSING
	//================================================================================80

	// write_graphics(	"image_out.pgm",
	// 				image,
	// 				Nr,
	// 				Nc,
	// 				1,
	// 				255);

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

	// cudaFree(dummy);

	cudaFree(image_ori);
	cudaFree(image);

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

	// bemps_free(tid);

	clock_gettime( CLOCK_REALTIME, &specific_time);
    now = localtime(&specific_time.tv_sec);
    millsec = specific_time.tv_nsec;

    millsec = floor (specific_time.tv_nsec/1.0e6);

    printf("After finish, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", 1900 + now->tm_year, 
        now->tm_mon + 1, now->tm_mday, now->tm_hour, 
        now->tm_min, now->tm_sec, millsec);

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

}

//====================================================================================================100
//	END OF FILE
//====================================================================================================100
