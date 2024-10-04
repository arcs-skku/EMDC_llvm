

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>

// includes, kernels
#include "backprop_cuda_kernel.cu"
#include "backprop.h"

#include <chrono>
#include <iostream>
using namespace std::chrono;

////////////////////////////////////////////////////////////////////////////////

extern "C"
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern "C"
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern "C"
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern "C" 
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);


extern "C"
int setup(int argc, char** argv);

extern "C"
float **alloc_2d_dbl(int m, int n);

extern "C"
float squash(float x);

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

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
	setup(argc, argv);
}


extern "C"
void bpnn_train_cuda(BPNN *net, float *eo, float *eh, int os_perc)
{
  int in, hid, out;
  float out_err, hid_err;
  
  // size_t mem_req = 1508900864;

  // size_t k1_mem_req = 2420113480;
  // size_t k2_mem_req = 4708106240;

  size_t k1_mem_req = 4836032584;
  size_t k2_mem_req = 9405726720;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
  printf("In: %d, Hid: %d, Out: %d\n", in, hid, out);
#ifdef GPU  
  int m = 0;
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float sum;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  num_blocks = in / 16;  
  int dim = 1;
  if (num_blocks >= 65536) {
    dim = num_blocks / 32768;
    num_blocks = 32768;
  }
  
  dim3  grid( dim , num_blocks);
  dim3  threads(16 , 16);

  printf("OS_perc: %d\n", os_perc);

  size_t f_free_mem, f_total_mem;
  size_t free_mem, total_mem;
  CUDA_CHECK(cudaMemGetInfo(&f_free_mem, &f_total_mem));

  printf("First Free: %zd, Total: %zd\n", f_free_mem, f_total_mem);

  float dummy_perc = ((float)os_perc / (float)100);
  printf("Dummy percentage: %f\n", dummy_perc);

  int* dummy;
  size_t dummy_size = page_align(f_free_mem * dummy_perc) - page_align(k1_mem_req);
  printf("Dummy size: %zd\n", dummy_size);
  
  CUDA_CHECK(cudaMalloc((void**) &dummy, dummy_size))
  CUDA_CHECK(cudaMemset(dummy, 0, dummy_size));

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

  input_weights_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_prev_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));
 
  CUDA_CHECK(cudaMallocManaged(&input_cuda, (in + 1) * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&output_hidden_cuda, (hid + 1) * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&hidden_partial_sum, num_blocks * WIDTH * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float)));
  memcpy(input_cuda,net->input_units, (in + 1)  *sizeof(float));

  // this preprocessing stage is added to correct the bugs of wrong memcopy using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++) {	
   for (int j = 0; j <= hid; j++) {
      input_hidden_cuda[m] = net->input_weights[k][j];
      input_prev_weights_cuda[m] = net-> input_prev_weights[k][j];
      m++;
    }
  }
  
  for(int k = 0; k < in + 1; k++){
    input_cuda[k] = 0;
  }

  for(int k = 0; k < hid + 1; k++){
    output_hidden_cuda[k] = 0;
  }

  for(int k = 0; k < num_blocks * WIDTH; k++){
    hidden_partial_sum[k] = 0;
  }

#endif

#ifdef CPU

  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);

#endif

#ifdef GPU
 
  printf("Performing GPU computation\n");

// CUDA_CHECK(cudaMemPrefetchAsync(input_cuda,(in + 1) * sizeof(float), 0, 0));
// CUDA_CHECK(cudaMemPrefetchAsync(output_hidden_cuda,(hid + 1) * sizeof(float), 0, 0));
// CUDA_CHECK(cudaMemPrefetchAsync(input_hidden_cuda,(in + 1) * (hid + 1) * sizeof(float), 0, 0));
// CUDA_CHECK(cudaMemPrefetchAsync(hidden_partial_sum,num_blocks * WIDTH * sizeof(float), 0, 0));
// CUDA_CHECK(cudaMemPrefetchAsync(input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float), 0, 0));

auto t_start = high_resolution_clock::now();

// CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

// printf("Before First Kernel Free: %zd, Total: %zd\n", free_mem, total_mem);

cudaEvent_t l_start, l_stop;
CUDA_CHECK(cudaEventCreate(&l_start));
CUDA_CHECK(cudaEventCreate(&l_stop));

float total_l, time;

CUDA_CHECK(cudaEventRecord(l_start));

bpnn_layerforward_CUDA<<< grid, threads >>>(input_cuda,
	                                          output_hidden_cuda,
											  input_hidden_cuda,
											  hidden_partial_sum,
											  in,
											  hid);

CUDA_CHECK(cudaEventRecord(l_stop));
CUDA_CHECK(cudaEventSynchronize(l_stop));
CUDA_CHECK(cudaEventElapsedTime(&total_l, l_start, l_stop));
										  
printf("First_kernel: %f\n", total_l);

auto t_stop = high_resolution_clock::now();
auto duration = duration_cast<milliseconds>(t_stop - t_start);
std::cout << "Kernel1 execution time: " << duration.count() << std::endl;

// CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

// printf("After First Kernel Free: %zd, Total: %zd\n", free_mem, total_mem);

for (int j = 1; j <= hid; j++) {
  sum = 0.0;
  for (int k = 0; k < num_blocks; k++) {	
    sum += hidden_partial_sum[k * hid + j-1] ;
  }
  sum += net->input_weights[0][j];
	net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
}

cudaFree(dummy);

dummy_size = page_align(f_free_mem * dummy_perc) - page_align(k2_mem_req);
printf("Dummy size: %zd\n", dummy_size);
  
CUDA_CHECK(cudaMalloc((void**) &dummy, dummy_size))
CUDA_CHECK(cudaMemset(dummy, 0, dummy_size));

#endif

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

#ifdef CPU

  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

#endif  


#ifdef GPU

CUDA_CHECK(cudaMallocManaged( &hidden_delta_cuda, (hid + 1) * sizeof(float)));
memcpy(hidden_delta_cuda, net->hidden_delta,(hid + 1)* sizeof(float));

// CUDA_CHECK(cudaMemPrefetchAsync(hidden_delta_cuda, (hid + 1) * sizeof(float), 0, 0));

// CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

// printf("Before Second Kernel Free: %zd, Total: %zd\n", free_mem, total_mem);

t_start = high_resolution_clock::now();

cudaEvent_t a_start, a_stop;
CUDA_CHECK(cudaEventCreate(&a_start));
CUDA_CHECK(cudaEventCreate(&a_stop));

float total_a;

CUDA_CHECK(cudaEventRecord(a_start));

  bpnn_adjust_weights_cuda<<< grid, threads >>>(hidden_delta_cuda,  
												hid, 
												input_cuda, 
												in,
												input_hidden_cuda, 
												input_prev_weights_cuda
												);

CUDA_CHECK(cudaEventRecord(a_stop));
CUDA_CHECK(cudaEventSynchronize(a_stop));
CUDA_CHECK(cudaEventElapsedTime(&total_a, a_start, a_stop));
										  
printf("Second_kernel: %f\n", total_a);

t_stop = high_resolution_clock::now();
duration = duration_cast<milliseconds>(t_stop - t_start);
std::cout << "Kernel2 execution time: " << duration.count() << std::endl;

// CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

// printf("After Second Kernel Free: %zd, Total: %zd\n", free_mem, total_mem);

  memcpy(net->input_units, input_cuda, (in + 1)  * sizeof(float));

  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);
  cudaFree(input_prev_weights_cuda);
  cudaFree(hidden_delta_cuda);
  cudaFree(dummy);
  
CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

printf("Last Free: %zd, Total: %zd\n", free_mem, total_mem);

  clock_gettime( CLOCK_REALTIME, &specific_time);
  now = localtime(&specific_time.tv_sec);
  millsec = specific_time.tv_nsec;

  millsec = floor (specific_time.tv_nsec/1.0e6);

  printf("Task finish, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", 1900 + now->tm_year, 
    now->tm_mon + 1, now->tm_mday, now->tm_hour, 
    now->tm_min, now->tm_sec, millsec);

  // free(partial_sum);
  // free(input_weights_one_dim);
  // free(input_weights_prev_one_dim);

#endif   
  
  
  

}
