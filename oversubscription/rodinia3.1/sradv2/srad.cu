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

#include <chrono>
#include <iostream>

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
	struct timespec specific_time;
	struct tm *now;
	int millsec;
	clock_gettime( CLOCK_REALTIME, &specific_time);
	now = localtime(&specific_time.tv_sec);
	millsec = specific_time.tv_nsec;
  
	millsec = floor (specific_time.tv_nsec/1.0e6);
  
	printf("Application begin, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", 1900 + now->tm_year, 
		now->tm_mon + 1, now->tm_mday, now->tm_hour, 
		now->tm_min, now->tm_sec, millsec);


  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
    runTest( argc, argv);

	clock_gettime( CLOCK_REALTIME, &specific_time);
	now = localtime(&specific_time.tv_sec);
	millsec = specific_time.tv_nsec;
  
	millsec = floor (specific_time.tv_nsec/1.0e6);
  
	printf("Application end, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", 1900 + now->tm_year, 
		now->tm_mon + 1, now->tm_mday, now->tm_hour, 
		now->tm_min, now->tm_sec, millsec);
		
    return EXIT_SUCCESS;
}

void
runTest( int argc, char** argv) 
{
    int rows, cols, size_I, size_R, niter = 10, iter;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI,varROI ;

	int ef_cnt = 0;
  	int ret_dev_id = 0;
	int os_perc = 0;

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
		os_perc = atoi(argv[9]);
	}
    else{
	usage(argc, argv);
    }



	size_I = cols * rows;
    size_R = (r2-r1+1)*(c2-c1+1);   

	I = (float *)malloc( size_I * sizeof(float) );
    // J = (float *)malloc( size_I * sizeof(float) );
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
	
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(block_x , block_y);

	size_t mem_req = 6442450944;

	printf("OS_perc: %d\n", os_perc);

  	size_t free_mem, total_mem;
	CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

	printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

	float dummy_perc = ((float)os_perc / (float)100);
	printf("Dummy percentage: %f\n", dummy_perc);

	int* dummy;
	size_t dummy_size = page_align(free_mem * dummy_perc) - page_align(mem_req);
	printf("Dummy size: %zd\n", dummy_size);

	CUDA_CHECK(cudaMalloc((void**) &dummy, dummy_size))
	CUDA_CHECK(cudaMemset(dummy, 0, dummy_size));

	//Allocate device memory
	CUDA_CHECK(cudaMallocManaged(&J_cuda, sizeof(float)* size_I));
	CUDA_CHECK(cudaMallocManaged(&C_cuda, sizeof(float)* size_I));
	CUDA_CHECK(cudaMallocManaged(&E_C, sizeof(float)* size_I));
	CUDA_CHECK(cudaMallocManaged(&W_C, sizeof(float)* size_I));
	CUDA_CHECK(cudaMallocManaged(&S_C, sizeof(float)* size_I));
	CUDA_CHECK(cudaMallocManaged(&N_C, sizeof(float)* size_I));
	
#endif 

	printf("Randomizing the input matrix\n");
	//Generate a random matrix
	random_matrix(I, rows, cols);

    for (int k = 0;  k < size_I; k++ ) {
     	J_cuda[k] = (float)exp(I[k]) ;
    }
	printf("Start the SRAD main loop\n");
	
	float total_k1, total_k2;

	cudaEvent_t k1_start, k1_stop, k2_start, k2_stop;

 for (iter=0; iter< niter; iter++){     
		sum=0; sum2=0;
        for (int i=r1; i<=r2; i++) {
            for (int j=c1; j<=c2; j++) {
                tmp   = J_cuda[i * cols + j];
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
	auto t_start = high_resolution_clock::now();
	
	CUDA_CHECK(cudaEventCreate(&k1_start));
	CUDA_CHECK(cudaEventCreate(&k1_stop));
	CUDA_CHECK(cudaEventRecord(k1_start));

	// CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

	// printf("Free: %zd, Total: %zd\n", free_mem, total_mem);
	
	//Run kernels
	srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr); 

	CUDA_CHECK(cudaEventRecord(k1_stop));
	CUDA_CHECK(cudaEventSynchronize(k1_stop));
	CUDA_CHECK(cudaEventElapsedTime(&total_k1, k1_start, k1_stop));

	printf("Kernel1: %f\n", total_k1);

	// CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

	// printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

	CUDA_CHECK(cudaEventCreate(&k2_start));
	CUDA_CHECK(cudaEventCreate(&k2_stop));
	CUDA_CHECK(cudaEventRecord(k2_start));

	srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr); 

	CUDA_CHECK(cudaEventRecord(k2_stop));
	CUDA_CHECK(cudaEventSynchronize(k2_stop));
	CUDA_CHECK(cudaEventElapsedTime(&total_k2, k2_start, k2_stop));

	printf("Kernel2: %f\n", total_k2);

	auto t_stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Kernel execution time: " << duration.count() << std::endl;

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

   	CUDA_CHECK(cudaFree(dummy));

    CUDA_CHECK(cudaFree(C_cuda));
	CUDA_CHECK(cudaFree(J_cuda));
	CUDA_CHECK(cudaFree(E_C));
	CUDA_CHECK(cudaFree(W_C));
	CUDA_CHECK(cudaFree(N_C));
	CUDA_CHECK(cudaFree(S_C));

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

