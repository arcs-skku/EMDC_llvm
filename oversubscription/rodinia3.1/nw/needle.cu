#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "needle.h"
#include <cuda.h>
#include <sys/time.h>

// includes, kernels
#include "needle_kernel.cu"

#include <fcntl.h>
#include <error.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>

#include <chrono>

using namespace std::chrono;
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

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{

  printf("WG size of kernel = %d \n", BLOCK_SIZE);

    runTest( argc, argv);

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
	
	size_t mem_req = 8724152320;

    int os_perc;

    // the lengths of the two sequences should be able to divided by 16.
	// And at current stage  max_rows needs to equal max_cols
	if (argc == 4)
	{
		max_rows = atoi(argv[1]);
		max_cols = atoi(argv[1]);
		penalty = atoi(argv[2]);
		os_perc = atoi(argv[3]);
	}
    else{
	usage(argc, argv);
    }
	
	if(atoi(argv[1])%16!=0){
	fprintf(stderr,"The dimension values must be a multiple of 16\n");
	exit(1);
	}
	
	printf("OS_perc: %d\n", os_perc);

  	size_t free_mem, total_mem;
	CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

	printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

	float dummy_perc = ((float)os_perc / (float)100);
	printf("Dummy percentage: %f\n", dummy_perc);

	int* dummy;
	size_t dummy_size = page_align(free_mem * dummy_perc) - page_align(mem_req);
	printf("Dummy size: %zd\n", dummy_size);

	// CUDA_CHECK(cudaMalloc((void**) &dummy, dummy_size))
	// CUDA_CHECK(cudaMemset(dummy, 0, dummy_size));

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
	size = max_cols * max_rows;

	// referrence = (int *)malloc( max_rows * max_cols * sizeof(int) );
    // input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
	// output_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
	struct timespec specific_time;
	struct tm *now;
	int millsec;
	clock_gettime( CLOCK_REALTIME, &specific_time);
	now = localtime(&specific_time.tv_sec);
	millsec = specific_time.tv_nsec;

	millsec = floor (specific_time.tv_nsec/1.0e6);

	printf("Task start, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", 1900 + now->tm_year, 
		now->tm_mon + 1, now->tm_mday, now->tm_hour, 
		now->tm_min, now->tm_sec, millsec);

	CUDA_CHECK(cudaMallocManaged(&referrence_cuda, sizeof(int)*size));
	CUDA_CHECK(cudaMallocManaged(&input_itemsets, sizeof(int)*size));

	if (!input_itemsets)
		fprintf(stderr, "error: can not allocate memory");

    srand ( 7 );
	
	
    for (int i = 0 ; i < max_cols; i++){
		for (int j = 0 ; j < max_rows; j++){
			input_itemsets[i*max_cols+j] = 0;
		}
	}
	
	printf("Start Needleman-Wunsch\n");
	
	for( int i=1; i< max_rows ; i++){    //please define your own sequence. 
       input_itemsets[i*max_cols] = rand() % 10 + 1;
	}
    for( int j=1; j< max_cols ; j++){    //please define your own sequence.
       input_itemsets[j] = rand() % 10 + 1;
	}


	for (int i = 1 ; i < max_cols; i++){
		for (int j = 1 ; j < max_rows; j++){
		referrence_cuda[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
		}
	}

    for( int i = 1; i< max_rows ; i++)
       input_itemsets[i*max_cols] = -i * penalty;
	for( int j = 1; j< max_cols ; j++)
       input_itemsets[j] = -j * penalty;


    // size = max_cols * max_rows;
	// cudaMalloc((void**)& referrence_cuda, sizeof(int)*size);
	// cudaMalloc((void**)& matrix_cuda, sizeof(int)*size);
	
	// cudaMemcpy(referrence_cuda, referrence, sizeof(int) * size, cudaMemcpyHostToDevice);
	// cudaMemcpy(matrix_cuda, input_itemsets, sizeof(int) * size, cudaMemcpyHostToDevice);

    dim3 dimGrid;
	dim3 dimBlock(BLOCK_SIZE, 1);
	int block_width = ( max_cols - 1 )/BLOCK_SIZE;

	printf("Processing top-left matrix\n");
	//process top-left matrix

	// CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

	// printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

	cudaEvent_t *k1_start, *k1_stop, *k2_start, *k2_stop;
	k1_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 2049);
	k1_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 2049);
	k2_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 2047);
	k2_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 2047);
	
	float total_k1, total_k2, time;
	total_k1 = 0;
	total_k2 = 0;

	auto t_start = high_resolution_clock::now();

	for( int i = 1 ; i <= block_width ; i++){
		
		CUDA_CHECK(cudaEventCreate(&k1_start[i]));
		CUDA_CHECK(cudaEventCreate(&k1_stop[i]));
		CUDA_CHECK(cudaEventRecord(k1_start[i]));

		dimGrid.x = i;
		dimGrid.y = 1;
		needle_cuda_shared_1<<<dimGrid, dimBlock>>>(referrence_cuda, input_itemsets
		                                      ,max_cols, penalty, i, block_width); 

		CUDA_CHECK(cudaEventRecord(k1_stop[i]));
	}
	printf("Processing bottom-right matrix\n");

	CUDA_CHECK(cudaEventSynchronize(k1_stop[2048]));

	auto t_stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Kernel execution time: " << duration.count() << std::endl;

	// CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

	// printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

	t_start = high_resolution_clock::now();

    //process bottom-right matrix
	for( int i = block_width - 1  ; i >= 1 ; i--){

		CUDA_CHECK(cudaEventCreate(&k2_start[2047 - i]));
		CUDA_CHECK(cudaEventCreate(&k2_stop[2047 - i]));
		CUDA_CHECK(cudaEventRecord(k2_start[2047 - i]));

		dimGrid.x = i;
		dimGrid.y = 1;
		needle_cuda_shared_2<<<dimGrid, dimBlock>>>(referrence_cuda, input_itemsets
		                                      ,max_cols, penalty, i, block_width); 

		CUDA_CHECK(cudaEventRecord(k2_stop[2047 - i]));
	}

	CUDA_CHECK(cudaEventSynchronize(k2_stop[2046]));

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Kernel execution time: " << duration.count() << std::endl;

	for(int i = 1; i < 2049; i++){
		cudaEventElapsedTime(&time, k1_start[i], k1_stop[i]);
		if(i < 10)
			printf("%f ", time);
		total_k1 += time;
	}

	printf("\n");

	for(int i = 0; i < 2047; i++){
		cudaEventElapsedTime(&time, k2_start[i], k2_stop[i]);
		if(i < 10)
			printf("%f ", time);
		total_k2 += time;
	}

	printf("\n");

	printf("K1_kernel: %f, k2_kernel %f\n", total_k1, total_k2);

	// cudaFree(dummy);

    // cudaMemcpy(output_itemsets, matrix_cuda, sizeof(int) * size, cudaMemcpyDeviceToHost);
	
//#define TRACEBACK
#ifdef TRACEBACK
	
FILE *fpo = fopen("result.txt","w");
fprintf(fpo, "print traceback value GPU:\n");

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

	cudaFree(referrence_cuda);
	cudaFree(input_itemsets);

	clock_gettime( CLOCK_REALTIME, &specific_time);
	now = localtime(&specific_time.tv_sec);
	millsec = specific_time.tv_nsec;

	millsec = floor (specific_time.tv_nsec/1.0e6);

	printf("Task finish, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", 1900 + now->tm_year, 
		now->tm_mon + 1, now->tm_mday, now->tm_hour, 
		now->tm_min, now->tm_sec, millsec);

	// free(referrence);
	// free(input_itemsets);
	// free(output_itemsets);
	
}
