#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(val) { \
	if (val != cudaSuccess) { \
		fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(val), __LINE__, __FILE__); \
		exit(val); \
	} \
}

__global__ void acc_mem (int* arr, unsigned int* clock_arr, int* chk_arr){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int clock_idx = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int start = clock();
    chk_arr[idx] = arr[idx* 1024];
    unsigned int stop = clock();
    
    clock_arr[clock_idx] = stop - start;
}

int main (int argc, char* argv[]){
    cudaFree(0);
    int *arr, *h_chk, *d_chk;
    unsigned int *h_clock_arr, *d_clock_arr;
    int page_size = 1024;
    int idx = 1;
    int clock_idx = 1;

    // for read
    h_chk = (int*) malloc (idx * sizeof(int));
    CUDA_CHECK(cudaMalloc(&d_chk, idx * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_chk, 0, idx * sizeof(int)));

    // for clock
    h_clock_arr = (unsigned int*) malloc (clock_idx * sizeof(unsigned int));
    CUDA_CHECK(cudaMalloc(&d_clock_arr, clock_idx * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_clock_arr, 0, clock_idx * sizeof(unsigned int)));

    CUDA_CHECK(cudaMallocManaged(&arr, idx * page_size * sizeof(int)));

    for(int i = 0; i < idx * page_size; i++){
        arr[i] = 1;
    }

    if(strcmp(argv[1], "GP") == 0){
        CUDA_CHECK(cudaMemAdvise(arr, idx * page_size * sizeof(int), cudaMemAdviseSetPreferredLocation, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(arr, idx * page_size * sizeof(int), 0, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    else if(strcmp(argv[1], "ZC") == 0){
        CUDA_CHECK(cudaMemAdvise(arr, idx * page_size * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        CUDA_CHECK(cudaMemAdvise(arr, idx * page_size * sizeof(int), cudaMemAdviseSetAccessedBy, 0));
    }
    else if(strcmp(argv[1], "RC") == 0){
        CUDA_CHECK(cudaMemAdvise(arr, idx * page_size * sizeof(int), cudaMemAdviseSetReadMostly, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(arr, idx * page_size * sizeof(int), 0, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    else if(strcmp(argv[1], "DP") == 0){
        CUDA_CHECK(cudaMemPrefetchAsync(arr, idx * page_size * sizeof(int), 0, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    acc_mem <<< 1, 1 >>>(arr, d_clock_arr, d_chk);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_chk, d_chk, idx * sizeof(int), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(h_clock_arr, d_clock_arr, clock_idx * sizeof(unsigned int), cudaMemcpyDefault));

    for(int i = 0; i < idx; i++){
        if(h_chk[i] != 1){
            printf("Err\n");
            break;
        }
    }
    
    for(int i = 0; i < clock_idx; i++){
        printf("%u\n", h_clock_arr[i]);
    }

    CUDA_CHECK(cudaFree(d_chk));
    CUDA_CHECK(cudaFree(d_clock_arr));
    CUDA_CHECK(cudaFree(arr));
}