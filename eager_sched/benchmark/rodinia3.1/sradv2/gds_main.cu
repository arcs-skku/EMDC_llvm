// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "srad.h"

// includes, project
#include <cuda.h>

// includes, kernels
#include "srad_kernel.cu"

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

void random_matrix(float *I, int rows, int cols);
void runTest( int argc, char** argv);
void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lamda> <no. of iter>\n", argv[0]);
	fprintf(stderr, "\t<rows>   - number of rows\n");
	fprintf(stderr, "\t<cols>    - number of cols\n");
	fprintf(stderr, "\t<y1> 	 - y1 value of the speckle\n");
	fprintf(stderr, "\t<y2>      - y2 value of the speckle\n");
	fprintf(stderr, "\t<x1>       - x1 value of the speckle\n");
	fprintf(stderr, "\t<x2>       - x2 value of the speckle\n");
	fprintf(stderr, "\t<lamda>   - lambda (0,1)\n");
	fprintf(stderr, "\t<no. of iter>   - number of iterations\n");
	
	exit(1);
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
	int tid = atoi(argv[9]);

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


  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
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

void
runTest( int argc, char** argv) 
{
    int rows, cols, size_I, size_R, niter = 10, iter;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI,varROI ;

	size_t ef_mem;
  	int ef_cnt = 0;
  	int ret_dev_id;
	int tid;

	Parameters ret1;
	Parameters ret2;
	Parameters ret3;
	Parameters ret4;
	Parameters ret5;
	Parameters ret6;

#ifdef CPU
	float Jc, G2, L, num, den, qsqr;
	int *iN,*iS,*jE,*jW, k;
	float *dN,*dS,*dW,*dE;
	float cN,cS,cW,cE,D;
#endif

#ifdef GPU
	
	float *J_cuda;
    float *C_cuda;
	float *E_C, *W_C, *N_C, *S_C;

#endif

	unsigned int r1, r2, c1, c2;
	float *c;
    
	
 
	if (argc == 10)
	{
		rows = atoi(argv[1]);  //number of rows in the domain
		cols = atoi(argv[2]);  //number of cols in the domain
		if ((rows%16!=0) || (cols%16!=0)){
		fprintf(stderr, "rows and cols must be multiples of 16\n");
		exit(1);
		}
		r1   = atoi(argv[3]);  //y1 position of the speckle
		r2   = atoi(argv[4]);  //y2 position of the speckle
		c1   = atoi(argv[5]);  //x1 position of the speckle
		c2   = atoi(argv[6]);  //x2 position of the speckle
		lambda = atof(argv[7]); //Lambda value
		niter = atoi(argv[8]); //number of iterations
		tid = atoi(argv[9]);
	}
    else{
	usage(argc, argv);
    }



	size_I = cols * rows;
    size_R = (r2-r1+1)*(c2-c1+1);   

	I = (float *)malloc( size_I * sizeof(float) );
    J = (float *)malloc( size_I * sizeof(float) );
	c  = (float *)malloc(sizeof(float)* size_I) ;


#ifdef CPU

    iN = (int *)malloc(sizeof(unsigned int*) * rows) ;
    iS = (int *)malloc(sizeof(unsigned int*) * rows) ;
    jW = (int *)malloc(sizeof(unsigned int*) * cols) ;
    jE = (int *)malloc(sizeof(unsigned int*) * cols) ;    


	dN = (float *)malloc(sizeof(float)* size_I) ;
    dS = (float *)malloc(sizeof(float)* size_I) ;
    dW = (float *)malloc(sizeof(float)* size_I) ;
    dE = (float *)malloc(sizeof(float)* size_I) ;    
    

    for (int i=0; i< rows; i++) {
        iN[i] = i-1;
        iS[i] = i+1;
    }    
    for (int j=0; j< cols; j++) {
        jW[j] = j-1;
        jE[j] = j+1;
    }
    iN[0]    = 0;
    iS[rows-1] = rows-1;
    jW[0]    = 0;
    jE[cols-1] = cols-1;

#endif

#ifdef GPU

	int block_x = cols/BLOCK_SIZE ;
	int block_y = rows/BLOCK_SIZE ;
	
	// printf("BLOCK_SIZE: %d, block_x: %d, block_y: %d\n", BLOCK_SIZE, block_x, block_y);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(block_x , block_y);

	size_t membytes = 0;
	membytes += page_align(sizeof(float)* size_I);
	membytes += page_align(sizeof(float)* size_I);
	membytes += page_align(sizeof(float)* size_I);
	membytes += page_align(sizeof(float)* size_I);
	membytes += page_align(sizeof(float)* size_I);
	membytes += page_align(sizeof(float)* size_I);
  	membytes += 309 * 1024 * 1024;
	
#endif 

	printf("Randomizing the input matrix\n");
	//Generate a random matrix
	random_matrix(I, rows, cols);

    for (int k = 0;  k < size_I; k++ ) {
     	J[k] = (float)exp(I[k]) ;
    }
	// CUfileError_t status;
    // CUfileDescr_t cf_descr;
    // CUfileHandle_t cf_handle;
	// int fd = -1;
	// ssize_t ret = -1;

	// status = cuFileDriverOpen();

	// if(rows == 8192){
	// 	printf("Small\n");
	// 	fd = open("small/J_small.txt", O_RDONLY | O_DIRECT, 0644);
	// 	memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
	// 	cf_descr.handle.fd = fd;
	// 	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
	// 	status = cuFileHandleRegister(&cf_handle, &cf_descr);
	// 	ret = -1;
 
	// 	ret = cuFileRead(cf_handle, J_cuda, size_I, 0, 0);
	// 	cuFileHandleDeregister(cf_handle);
    //  	close(fd);
	// }
	// else if(rows == 16384){
	// 	printf("Large\n");
	// 	fd = open("large/J_large.txt", O_RDONLY | O_DIRECT, 0644);
	// 	memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
	// 	cf_descr.handle.fd = fd;
	// 	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
	// 	status = cuFileHandleRegister(&cf_handle, &cf_descr);
	// 	ret = -1;
 
	// 	ret = cuFileRead(cf_handle, J_cuda, size_I, 0, 0);
	// 	cuFileHandleDeregister(cf_handle);
    //  	close(fd);
	// }
	
	// memset(C_cuda, 0, size_I * sizeof(float));
	// memset(E_C, 0, size_I * sizeof(float));
	// memset(W_C, 0, size_I * sizeof(float));
	// memset(S_C, 0, size_I * sizeof(float));
	// memset(N_C, 0, size_I * sizeof(float));

	printf("Start the SRAD main loop\n");
	
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
		
	long orig_alloc_mem = bemps_begin(tid, dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, membytes, ret_dev_id);
		
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

	auto t_start = high_resolution_clock::now();

	//Allocate device memory
	CUDA_CHECK(cudaMallocManaged(&J_cuda, sizeof(float)* size_I));
	CUDA_CHECK(cudaMallocManaged(&C_cuda, sizeof(float)* size_I));
	CUDA_CHECK(cudaMallocManaged(&E_C, sizeof(float)* size_I));
	CUDA_CHECK(cudaMallocManaged(&W_C, sizeof(float)* size_I));
	CUDA_CHECK(cudaMallocManaged(&S_C, sizeof(float)* size_I));
	CUDA_CHECK(cudaMallocManaged(&N_C, sizeof(float)* size_I));

	auto t_stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " First device mem allocation time: " << duration.count() << std::endl;

	ret1.devPtr = J_cuda;
	ret1.advice = cudaMemAdviseSetPreferredLocation;
	ret1.device = ret_dev_id;
	ret1.alloc_size = size_I * sizeof(float);

  	ret2.devPtr = C_cuda;
	ret2.advice = cudaMemAdviseSetPreferredLocation;
	ret2.device = ret_dev_id;
	ret2.alloc_size = size_I * sizeof(float);

  	ret3.devPtr = E_C;
	ret3.advice = cudaMemAdviseSetPreferredLocation;
	ret3.device = ret_dev_id;
	ret3.alloc_size = size_I * sizeof(float);

  	ret4.devPtr = W_C;
	ret4.advice = cudaMemAdviseSetPreferredLocation;
	ret4.device = ret_dev_id;
	ret4.alloc_size = size_I * sizeof(float);

  	ret5.devPtr = S_C;
	ret5.advice = cudaMemAdviseSetPreferredLocation;
	ret5.device = ret_dev_id;
	ret5.alloc_size = size_I * sizeof(float);

	ret6.devPtr = N_C;
	ret6.advice = cudaMemAdviseSetPreferredLocation;
	ret6.device = ret_dev_id;
	ret6.alloc_size = size_I * sizeof(float);

	mem_list.push_back(ret1);
	mem_list.push_back(ret2);
	mem_list.push_back(ret3);
	mem_list.push_back(ret4);
	mem_list.push_back(ret5);
	mem_list.push_back(ret6);

	if(full == 1){
		// CUDA_CHECK(cudaMemPrefetchAsync(ret1.devPtr, ret1.alloc_size, ret1.device, 0));
		for(Parameters var : mem_list){
			CUDA_CHECK(cudaMemAdvise(var.devPtr, var.alloc_size, var.advice, var.device));
		}
	}
	
	cudaEvent_t *s1_start, *s1_stop, *s2_start, *s2_stop;

	s1_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 2);
	s1_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 2);
	s2_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 2);
	s2_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 2);

	float total_s1, total_s2;

	total_s1 = 0;
	total_s2 = 0;

 for (iter=0; iter< niter; iter++){     
		sum=0; sum2=0;
        for (int i=r1; i<=r2; i++) {
            for (int j=c1; j<=c2; j++) {
                tmp   = J[i * cols + j];
                sum  += tmp ;
                sum2 += tmp*tmp;
            }
        }
        meanROI = sum / size_R;
        varROI  = (sum2 / size_R) - meanROI*meanROI;
        q0sqr   = varROI / (meanROI*meanROI);

#ifdef CPU
        
		for (int i = 0 ; i < rows ; i++) {
            for (int j = 0; j < cols; j++) { 
		
				k = i * cols + j;
				Jc = J[k];
 
				// directional derivates
                dN[k] = J[iN[i] * cols + j] - Jc;
                dS[k] = J[iS[i] * cols + j] - Jc;
                dW[k] = J[i * cols + jW[j]] - Jc;
                dE[k] = J[i * cols + jE[j]] - Jc;
			
                G2 = (dN[k]*dN[k] + dS[k]*dS[k] 
                    + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

   		        L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

				num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
                den  = 1 + (.25*L);
                qsqr = num/(den*den);
 
                // diffusion coefficent (equ 33)
                den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
                c[k] = 1.0 / (1.0+den) ;
                
                // saturate diffusion coefficent
                if (c[k] < 0) {c[k] = 0;}
                else if (c[k] > 1) {c[k] = 1;}
		}
	}
         for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {        

                // current index
                k = i * cols + j;
                
                // diffusion coefficent
					cN = c[k];
					cS = c[iS[i] * cols + j];
					cW = c[k];
					cE = c[i * cols + jE[j]];

                // divergence (equ 58)
                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
                
                // image update (equ 61)
                J[k] = J[k] + 0.25*lambda*D;
            }
	}

#endif // CPU
	// if(full == 1){
	// 	CUDA_CHECK(cudaMemPrefetchAsync(ret1.devPtr, ret1.alloc_size, ret1.device, 0));
	// 	for(Parameters ret : mem_list){
	// 		CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
	// 	}
	// }

#ifdef GPU

	t_start = high_resolution_clock::now();

	memcpy(J_cuda, J, sizeof(float) * size_I);

	if(full == 1){
		// CUDA_CHECK(cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemPrefetchAsync(J_cuda, sizeof(float) * size_I, ret_dev_id, 0));
	}

	// if(full == 1){
	// 	// CUDA_CHECK(cudaMemPrefetchAsync(ret1.devPtr, ret1.alloc_size, ret1.device, 0));
	// 	for(Parameters ret : mem_list){
	// 		CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
	// 	}
	// }

	// CUDA_CHECK(cudaDeviceSynchronize());

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Host to Device memcpy time: " << duration.count() << std::endl;

	cudaStream_t s1;
	CUDA_CHECK(cudaStreamCreateWithPriority(&s1, 0, priority));

	cudaEvent_t event1;
	CUDA_CHECK(cudaEventCreateWithFlags(&event1, cudaEventDisableTiming));

	CUDA_CHECK(cudaEventCreate(&s1_start[iter]));
	CUDA_CHECK(cudaEventCreate(&s1_stop[iter]));
	CUDA_CHECK(cudaEventRecord(s1_start[iter]));

	//Run kernels
	srad_cuda_1<<<dimGrid, dimBlock, 0, s1>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr); 

	CUDA_CHECK(cudaEventRecord(event1, s1));

	task_monitoring(event1, tid, orig_alloc_mem, membytes);
				
	CUDA_CHECK(cudaEventRecord(s1_stop[iter]));
		
	CUDA_CHECK(cudaStreamSynchronize(s1));

	CUDA_CHECK(cudaStreamDestroy(s1));
	CUDA_CHECK(cudaEventDestroy(event1));

	cudaStream_t s2;
	CUDA_CHECK(cudaStreamCreateWithPriority(&s2, 0, priority));

	cudaEvent_t event2;
	CUDA_CHECK(cudaEventCreateWithFlags(&event2, cudaEventDisableTiming));

	CUDA_CHECK(cudaEventCreate(&s2_start[iter]));
	CUDA_CHECK(cudaEventCreate(&s2_stop[iter]));
	CUDA_CHECK(cudaEventRecord(s2_start[iter]));

	srad_cuda_2<<<dimGrid, dimBlock, 0, s2>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr); 

	CUDA_CHECK(cudaEventRecord(event2, s2));

	task_monitoring(event2, tid, orig_alloc_mem, membytes);
				
	CUDA_CHECK(cudaEventRecord(s2_stop[iter]));
		
	CUDA_CHECK(cudaStreamSynchronize(s2));

	CUDA_CHECK(cudaStreamDestroy(s2));
	CUDA_CHECK(cudaEventDestroy(event2));

	if(iter == 0){
	//Copy data from device memory to main memory
		t_start = high_resolution_clock::now();

		CUDA_CHECK(cudaMemPrefetchAsync(J_cuda, sizeof(float) * size_I, cudaCpuDeviceId, 0));
		memcpy(J, J_cuda, sizeof(float) * size_I);

		t_stop = high_resolution_clock::now();
		duration = duration_cast<milliseconds>(t_stop - t_start);
		std::cout << "Device to Host memcpy time: " << duration.count() << std::endl;

	// CUDA_CHECK(cudaMemPrefetchAsync(ret1.devPtr, ret1.alloc_size, cudaCpuDeviceId, 0));
	}
#endif   
}

#ifdef OUTPUT
    //Printing output	
		printf("Printing Output:\n"); 
    for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
         printf("%.5f ", J_cuda[i * cols + j]); 
		}	
     printf("\n"); 
   }
#endif 

	printf("Computation Done\n");

	free(I);
	// free(J);
#ifdef CPU
	free(iN); free(iS); free(jW); free(jE);
    free(dN); free(dS); free(dW); free(dE);
#endif
#ifdef GPU

	CUDA_CHECK(cudaEventSynchronize(s1_stop[1]));
	CUDA_CHECK(cudaEventSynchronize(s2_stop[1]));

	float time;

	for(int i = 0; i < 2; i++){
		cudaEventElapsedTime(&time, s1_start[i], s1_stop[i]);
		// printf("%f ", time);
		total_s1 += time;
		cudaEventElapsedTime(&time, s2_start[i], s2_stop[i]);
		// printf("%f\n", time);
		total_s2 += time;
	}

	printf("Fisrt_kernel: %f, Second_kernel %f\n", total_s1, total_s2);

// CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

//   printf("Free: %zd, Total: %zd\n", free_mem, total_mem);
  
  	CUDA_CHECK(cudaFree(E_C));
	  CUDA_CHECK(cudaFree(W_C));
	  CUDA_CHECK(cudaFree(N_C));
	  CUDA_CHECK(cudaFree(S_C));
	  CUDA_CHECK(cudaFree(C_cuda));

//   CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

//   printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

  if(full == 1){
    ef_mem = page_align(sizeof(float)* size_I) * 5;
    pre_bemps_free(tid, ef_mem);
    ef_cnt = 1;
  }

t_start = high_resolution_clock::now();

char* str1 = argv[9];
	char* str2 = "_output.txt";
	strcat(str1, str2);

	CUfileError_t status;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
	int fd = -1;
	ssize_t ret = -1;

	status = cuFileDriverOpen();

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

	ret = cuFileWrite(cf_handle, ret1.devPtr, ret1.alloc_size, 0, 0);
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

	t_start = high_resolution_clock::now();

	CUDA_CHECK(cudaFree(J_cuda));

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

#endif 
	free(c);
  
}


void random_matrix(float *I, int rows, int cols){
    
	srand(7);
	
	for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
		 I[i * cols + j] = rand()/(float)RAND_MAX ;
		}
	}

}

