

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
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
// #include <string>

using namespace std::chrono;

// includes, kernels
#include "backprop_cuda_kernel.cu"
#include "backprop.h"

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

void reverse(char s[])
 {
     int i, j;
     char c;
 
     for (i = 0, j = strlen(s)-1; i<j; i++, j--) {
         c = s[i];
         s[i] = s[j];
         s[j] = c;
     }
 }

void itoa(int n, char s[])
 {
     int i, sign;
 
     if ((sign = n) < 0)  /* record sign */
         n = -n;          /* make n positive */
     i = 0;
     do {       /* generate digits in reverse order */
         s[i++] = n % 10 + '0';   /* get next digit */
     } while ((n /= 10) > 0);     /* delete it */
     if (sign < 0)
         s[i++] = '-';
     s[i] = '\0';
     reverse(s);
 }

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

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
	int tid = atoi(argv[2]);
  printf("TID: %d\n", tid);

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

  // int tid = atoi(argv[2]);
  // printf("TID: %d\n", tid);
	setup(argc, argv);

  clock_gettime( CLOCK_REALTIME, &specific_time);
  now = localtime(&specific_time.tv_sec);
  millsec = specific_time.tv_nsec;

  millsec = floor (specific_time.tv_nsec/1.0e6);

  printf("TID: %d Application end, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
    now->tm_mon + 1, now->tm_mday, now->tm_hour, 
    now->tm_min, now->tm_sec, millsec);
}


extern "C"
void bpnn_train_cuda(BPNN *net, float *eo, float *eh, int tid)
{
	// int tid = atoi(argv[2]);
  int in, hid, out;
  float out_err, hid_err;
  
  size_t ef_mem;
  int ef_cnt = 0;
  int ret_dev_id;

  Parameters ret1;
	Parameters ret2;
	Parameters ret3;
	Parameters ret4;
	Parameters ret5;
	Parameters ret6;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
  printf("In: %d, Hid: %d, Out: %d\n", in, hid, out); // hid, out은 고정 16, 1
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

  float *tmp_input_hidden_cuda;
  float *tmp_output_hidden_cuda;
  float *tmp_hidden_partial_sum;
  float *tmp_input_prev_weights_cuda;

  num_blocks = in / 16;  
  int dim = 1;
  if (num_blocks >= 65536) {
    dim = num_blocks / 32768;
    num_blocks = 32768;
  }
  
  tmp_input_hidden_cuda = (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
//   tmp_input_prev_weights_cuda = (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
// //   tmp_output_hidden_cuda = (float *)malloc((hid + 1) * sizeof(float));
//   tmp_hidden_partial_sum = (float *)malloc(num_blocks * WIDTH * sizeof(float));
  

//   for (int k = 0; k <= in; k++) {	
// 	for (int j = 0; j <= hid; j++) {
// 	   tmp_input_hidden_cuda[m] = net->input_weights[k][j];
// 	   tmp_input_prev_weights_cuda[m] = net-> input_prev_weights[k][j];
// 	   m++;
// 	 }
//    }

  dim3  grid( dim , num_blocks);
  dim3  threads(16 , 16);

  size_t membytes = 0;
	membytes += page_align((in + 1) * sizeof(float));
	membytes += page_align((hid + 1) * sizeof(float));
	membytes += page_align((in + 1) * (hid + 1) * sizeof(float));
	membytes += page_align(num_blocks * WIDTH * sizeof(float));
	// membytes += page_align((hid + 1) * sizeof(float));
	membytes += page_align((in + 1) * (hid + 1) * sizeof(float));
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

  long orig_alloc_mem = bemps_begin(tid, grid.x, grid.y, grid.z, threads.x, threads.y, threads.z, membytes, ret_dev_id);

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

	CUDA_CHECK(cudaMallocManaged(&input_cuda, (in + 1) * sizeof(float)));
	CUDA_CHECK(cudaMallocManaged(&output_hidden_cuda, (hid + 1) * sizeof(float)));
	CUDA_CHECK(cudaMallocManaged(&input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float)));
	CUDA_CHECK(cudaMallocManaged(&hidden_partial_sum, num_blocks * WIDTH * sizeof(float)));
	CUDA_CHECK(cudaMallocManaged(&input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float)));
	CUDA_CHECK(cudaMallocManaged( &hidden_delta_cuda, (hid + 1) * sizeof(float)));

  auto t_stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " First device mem allocation time: " << duration.count() << std::endl;

  ret1.devPtr = input_cuda;
	ret1.advice = cudaMemAdviseSetPreferredLocation;
	ret1.device = ret_dev_id;
	ret1.alloc_size = (in + 1) * sizeof(float);

  ret2.devPtr = output_hidden_cuda;
	ret2.advice = cudaMemAdviseSetPreferredLocation;
	ret2.device = ret_dev_id;
	ret2.alloc_size = (hid + 1) * sizeof(float);

  ret3.devPtr = input_hidden_cuda;
	ret3.advice = cudaMemAdviseSetPreferredLocation;
	ret3.device = ret_dev_id;
	ret3.alloc_size = (in + 1) * (hid + 1) * sizeof(float);

  ret4.devPtr = hidden_partial_sum;
	ret4.advice = cudaMemAdviseSetPreferredLocation;
	ret4.device = ret_dev_id;
	ret4.alloc_size = num_blocks * WIDTH * sizeof(float);

  ret5.devPtr = input_prev_weights_cuda;
	ret5.advice = cudaMemAdviseSetPreferredLocation;
	ret5.device = ret_dev_id;
	ret5.alloc_size = (in + 1) * (hid + 1) * sizeof(float);

	ret6.devPtr = hidden_delta_cuda;
ret6.advice = cudaMemAdviseSetPreferredLocation;
ret6.device = ret_dev_id;
ret6.alloc_size = (hid + 1) * sizeof(float);

  mem_list.push_back(ret1);
	mem_list.push_back(ret2);
	mem_list.push_back(ret3);
	mem_list.push_back(ret4);
	mem_list.push_back(ret5);
	mem_list.push_back(ret6);

  // host initializing
  t_start = high_resolution_clock::now();

  for (int k = 0; k <= in; k++) {	
	for (int j = 0; j <= hid; j++) {
	   tmp_input_hidden_cuda[m] = net->input_weights[k][j];
	   input_prev_weights_cuda[m] = net-> input_prev_weights[k][j];
	   m++;
	 }
   }

   t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Host initialization time: " << duration.count() << std::endl;

	t_start = high_resolution_clock::now();

	memcpy(input_cuda,net->input_units, (in + 1)  *sizeof(float));
	memcpy(input_hidden_cuda, tmp_input_hidden_cuda, ret3.alloc_size);

	if(full == 1){
		for(Parameters ret : mem_list){
			// CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, ret.device, 0));
			CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
		}
		// CUDA_CHECK(cudaMemcpy(input_cuda,net->input_units, (in + 1)  *sizeof(float), cudaMemcpyHostToDevice));
		// CUDA_CHECK(cudaMemcpy(input_hidden_cuda, tmp_input_hidden_cuda, ret3.alloc_size, cudaMemcpyHostToDevice));
		// CUDA_CHECK(cudaMemPrefetchAsync(ret3.devPtr, ret3.alloc_size, ret3.device, 0));
		CUDA_CHECK(cudaMemPrefetchAsync(ret1.devPtr, ret1.alloc_size, ret1.device, 0));
		CUDA_CHECK(cudaMemPrefetchAsync(ret3.devPtr, ret3.alloc_size, ret3.device, 0));
		CUDA_CHECK(cudaMemPrefetchAsync(ret5.devPtr, ret5.alloc_size, ret5.device, 0));
	}

  t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Host to Device memcpy time: " << duration.count() << std::endl;

//   if(full == 1){
// 		for(Parameters ret : mem_list){
// 			// CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, ret.device, 0));
// 			CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
// 		}
// 	}

#endif

#ifdef CPU

  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);

#endif

#ifdef GPU

if(full == 0){
	el_wait(tid);
}

printf("Performing GPU computation\n");

cudaEvent_t l_start, l_stop;
CUDA_CHECK(cudaEventCreate(&l_start));
CUDA_CHECK(cudaEventCreate(&l_stop));

float total_l, time;

cudaStream_t s1;
CUDA_CHECK(cudaStreamCreateWithPriority(&s1, 0, priority));

cudaEvent_t event1;
CUDA_CHECK(cudaEventCreateWithFlags(&event1, cudaEventDisableTiming));

CUDA_CHECK(cudaEventRecord(l_start));

if(full == 1){
	nl_signal(tid);
}

bpnn_layerforward_CUDA<<< grid, threads, 0, s1 >>>(input_cuda,
	                                          output_hidden_cuda,
											  input_hidden_cuda,
											  hidden_partial_sum,
											  in,
											  hid);

CUDA_CHECK(cudaEventRecord(event1, s1));

task_monitoring(event1, tid, orig_alloc_mem, membytes);
										  
CUDA_CHECK(cudaEventRecord(l_stop));
CUDA_CHECK(cudaEventSynchronize(l_stop));
CUDA_CHECK(cudaEventElapsedTime(&total_l, l_start, l_stop));
										  
printf("First_kernel: %f\n", total_l);

t_start = high_resolution_clock::now();

// CUDA_CHECK(cudaMemcpy(tmp_hidden_partial_sum, hidden_partial_sum, num_blocks * WIDTH * sizeof(float), cudaMemcpyDeviceToHost));
CUDA_CHECK(cudaMemPrefetchAsync(hidden_partial_sum, num_blocks * WIDTH * sizeof(float), cudaCpuDeviceId, 0));
// CUDA_CHECK(cudaDeviceSynchronize());

t_stop = high_resolution_clock::now();
duration = duration_cast<milliseconds>(t_stop - t_start);
std::cout << "Tid: " << tid << " Device to Host memcpy time: " << duration.count() << std::endl;

for (int j = 1; j <= hid; j++) {
  sum = 0.0;
  for (int k = 0; k < num_blocks; k++) {	
    sum += hidden_partial_sum[k * hid + j-1] ;
  }
  sum += net->input_weights[0][j];
	net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
}

CUDA_CHECK(cudaFree(hidden_partial_sum));
CUDA_CHECK(cudaFree(output_hidden_cuda));

mem_list.erase(mem_list.begin()+3);
mem_list.erase(mem_list.begin()+1);

if(full == 1){
	ef_mem = page_align(num_blocks * WIDTH * sizeof(float));
    pre_bemps_free(tid, ef_mem);
    ef_cnt = 1;
}

#endif
	t_start = high_resolution_clock::now();

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

  t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Host processing time: " << duration.count() << std::endl;

#ifdef CPU

  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

#endif  


#ifdef GPU

// t_start = high_resolution_clock::now();

// CUDA_CHECK(cudaMallocManaged( &hidden_delta_cuda, (hid + 1) * sizeof(float)));

// t_stop = high_resolution_clock::now();
// duration = duration_cast<milliseconds>(t_stop - t_start);
// std::cout << "Tid: " << tid << " Second device mem allocation time: " << duration.count() << std::endl;

t_start = high_resolution_clock::now();

memcpy(hidden_delta_cuda, net->hidden_delta,(hid + 1)* sizeof(float));
memcpy(input_hidden_cuda, tmp_input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));	

if(full == 1){
	CUDA_CHECK(cudaMemPrefetchAsync(hidden_delta_cuda, (hid + 1)* sizeof(float), ret_dev_id, 0));
	CUDA_CHECK(cudaMemPrefetchAsync(input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float), ret_dev_id, 0));

	// CUDA_CHECK(cudaMemcpy(hidden_delta_cuda, net->hidden_delta,(hid + 1)* sizeof(float), cudaMemcpyHostToDevice));
	// CUDA_CHECK(cudaMemcpy(input_hidden_cuda, tmp_input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice));	
}

t_stop = high_resolution_clock::now();
duration = duration_cast<milliseconds>(t_stop - t_start);
std::cout << "Tid: " << tid << " Host to Device memcpy time: " << duration.count() << std::endl;

// if(full == 1){
//   CUDA_CHECK(cudaMemAdvise(ret6.devPtr, ret6.alloc_size, ret6.advice, ret6.device));
// //   CUDA_CHECK(cudaMemPrefetchAsync(ret6.devPtr, ret6.alloc_size, ret6.device, 0));  
// }

cudaEvent_t a_start, a_stop;
CUDA_CHECK(cudaEventCreate(&a_start));
CUDA_CHECK(cudaEventCreate(&a_stop));

float total_a;

cudaStream_t s2;
CUDA_CHECK(cudaStreamCreateWithPriority(&s2, 0, priority));

cudaEvent_t event2;
CUDA_CHECK(cudaEventCreateWithFlags(&event2, cudaEventDisableTiming));

CUDA_CHECK(cudaEventRecord(a_start));

bpnn_adjust_weights_cuda<<< grid, threads, 0, s2 >>>(hidden_delta_cuda,  
												hid, 
												input_cuda, 
												in,
												input_hidden_cuda, 
												input_prev_weights_cuda
												);

CUDA_CHECK(cudaEventRecord(event2, s2));

task_monitoring(event2, tid, orig_alloc_mem, membytes);
										  
CUDA_CHECK(cudaEventRecord(a_stop));
CUDA_CHECK(cudaEventSynchronize(a_stop));
CUDA_CHECK(cudaEventElapsedTime(&total_a, a_start, a_stop));
										  
printf("Second_kernel: %f\n", total_a);

CUDA_CHECK(cudaFree(input_prev_weights_cuda));
  CUDA_CHECK(cudaFree(hidden_delta_cuda));

  mem_list.erase(mem_list.begin()+3);
  mem_list.erase(mem_list.begin()+2);
  
//   for(Parameters test : mem_list){
// 	  printf("%zd\n", test.alloc_size);
//   }

  if(full == 1){
    if(ef_cnt == 1){
      ef_mem = page_align((in + 1) * (hid + 1) * sizeof(float)) + page_align((hid + 1) * sizeof(float));
			pre_bemps_free(tid, ef_mem);
		}
		else{
      ef_mem = page_align(num_blocks * WIDTH * sizeof(float)) + page_align((in + 1) * (hid + 1) * sizeof(float)) + page_align((hid + 1) * sizeof(float));
			pre_bemps_free(tid, ef_mem);
			ef_cnt = 1;
		}
  }

t_start = high_resolution_clock::now();

CUDA_CHECK(cudaMemPrefetchAsync(input_cuda, (in + 1) * sizeof(float), cudaCpuDeviceId, 0));
CUDA_CHECK(cudaMemPrefetchAsync(input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float), cudaCpuDeviceId, 0));

memcpy(net->input_units, input_cuda, (in + 1) * sizeof(float));
memcpy(tmp_input_hidden_cuda, input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));

t_stop = high_resolution_clock::now();
duration = duration_cast<milliseconds>(t_stop - t_start);
std::cout << "Tid: " << tid << " Device to Host memcpy time: " << duration.count() << std::endl;

t_start = high_resolution_clock::now();

CUDA_CHECK(cudaFree(input_cuda));
CUDA_CHECK(cudaFree(input_hidden_cuda));

t_stop = high_resolution_clock::now();
duration = duration_cast<milliseconds>(t_stop - t_start);
std::cout << "Tid: " << tid << " Device mem deallocation time: " << duration.count() << std::endl;

bemps_free(tid);

clock_gettime( CLOCK_REALTIME, &specific_time);
now = localtime(&specific_time.tv_sec);
millsec = specific_time.tv_nsec;

millsec = floor (specific_time.tv_nsec/1.0e6);

printf("TID: %d finish work, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
    now->tm_mon + 1, now->tm_mday, now->tm_hour, 
    now->tm_min, now->tm_sec, millsec);

t_start = high_resolution_clock::now();

size_t test_offset = 2147479552 / 4;

char str1[100];
itoa(tid, str1);
char* str2 = "_output.txt";
strcat(str1, str2);

int o_fd = -1;
o_fd = open(str1, O_CREAT | O_RDWR, 0664);
if (o_fd < 0) {
	std::cerr << "file open error:" << std::endl;
}

printf("%d\n", o_fd);

int cnt = 0;
size_t alloc_size = (in + 1) * sizeof(float);
int quo;
size_t rem;
ssize_t o_ret = -1;
while(1){
	quo = alloc_size / 2147479552;
	printf("quo: %d\n", quo);
	if(quo > 0){
		o_ret = write(o_fd, net->input_units+(cnt*test_offset), 2147479552);
		printf("%zd\n", o_ret);
		cnt++;
		alloc_size -= 2147479552;
		if (o_ret == -1) { 
			error(0,errno,"cannot access tmp.txt");
		}			
	}
	else{
		rem = alloc_size % 2147479552;
		o_ret = write(o_fd, net->input_units+(cnt*test_offset), rem);
		printf("%zd\n", o_ret);
		break;
	}
}

close(o_fd);
	
char h_str1[100];
itoa(tid, h_str1);
char* h_str2 = "_output_hidden.txt";
strcat(h_str1, h_str2);

o_fd = open(h_str1, O_CREAT | O_RDWR, 0664);
if (o_fd < 0) {
	std::cerr << "file open error:" << std::endl;
}

printf("%d\n", o_fd);

cnt = 0;
alloc_size = (in + 1) * (hid + 1) * sizeof(float);
o_ret = -1;
while(1){
	quo = alloc_size / 2147479552;
	printf("quo: %d\n", quo);
	if(quo > 0){
		o_ret = write(o_fd, tmp_input_hidden_cuda+(cnt*test_offset), 2147479552);
		printf("%zd\n", o_ret);
		cnt++;
		alloc_size -= 2147479552;
		if (o_ret == -1) { 
			error(0,errno,"cannot access tmp.txt");
		}			
	}
	else{
		rem = alloc_size % 2147479552;
		o_ret = write(o_fd, tmp_input_hidden_cuda+(cnt*test_offset), rem);
		printf("%zd\n", o_ret);
		break;
	}
}

close(o_fd);
t_stop = high_resolution_clock::now();
duration = duration_cast<milliseconds>(t_stop - t_start);
std::cout << "Tid: " << tid << " Host to Storage memcpy time: " << duration.count() << std::endl;


#endif     

}
