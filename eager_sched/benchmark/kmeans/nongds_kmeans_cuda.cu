#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <time.h>
#include <string>
#include <fcntl.h>
#include <unistd.h>

#include <bemps.hpp>

using namespace std::chrono;

bool full; // 1 = fully secured
int priority;
int sig_cnt = 0;
long u_mem;

struct Parameters{
  void* devPtr;
  size_t count;
  cudaMemoryAdvise advice;
  int device;
  size_t alloc_size; // 디바이스 메모리에 올릴 페이지 크기
  // std::bitset<kernel_num> bit; // liveness check
};

std::vector<Parameters> mem_list;

size_t page_align (size_t mem){
	if((mem % 2097152) != 0){
		return (2097152 * (mem / 2097152 + 1));
	}
	else{
		return mem;
	}
}

double gpu_time_used;
#define I(row, col, ncols) (row * ncols + col)

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}} 

#define CUDA_CHECK(val) { \
  if (val != cudaSuccess) { \
    fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(val), __LINE__, __FILE__); \
    exit(val); \
  } \
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
          priority = -5;
        }
        if(full == 1){
         cudaStream_t s_e;
         CUDA_CHECK(cudaStreamCreateWithPriority(&s_e, 0, priority));
         // CUDA_CHECK(cudaStreamCreate(&s_e));
         printf("Hello\n");
         int loop = 0;
         for(Parameters ret : mem_list){
           CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
           CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, ret.device, s_e));
          //  if(loop < 5){
          //    CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, ret.device, s_e));
          //  }
          //  loop++;
         }
         // CUDA_CHECK(cudaStreamSynchronize(s_e));
         // CUDA_CHECK(cudaStreamDestroy(s_e));
         break;
        }
      }
    }
  }
}

__global__ void get_dst(float *dst, float *x, float *y, 
			float *mu_x, float *mu_y){
  int i = blockIdx.x;
  int j = threadIdx.x;

  dst[I(i, j, blockDim.x)] = (x[i] - mu_x[j]) * (x[i] - mu_x[j]);
  dst[I(i, j, blockDim.x)] += (y[i] - mu_y[j]) * (y[i] - mu_y[j]); 
}

__global__ void regroup(int *group, float *dst, int k){
  int i = blockIdx.x;
  int j;
  float min_dst;
  
  min_dst = dst[I(i, 0, k)];
  group[i] = 1;

  for(j = 1; j < k; ++j){
    if(dst[I(i, j, k)] < min_dst){
      min_dst = dst[I(i, j, k)];
      group[i] = j + 1;
    }
  }
}

__global__ void clear(float *sum_x, float *sum_y, int *nx, int *ny){
  int j = threadIdx.x;
  
  sum_x[j] = 0;
  sum_y[j] = 0;
  nx[j] = 0;
  ny[j] = 0;
}

__global__ void recenter_step1(float *sum_x, float *sum_y, int *nx, int *ny,
			       float *x, float *y, int *group, int n){
  int i;
  // int j = threadIdx.x;

  int j = blockIdx.x * blockDim.x + threadIdx.x;

  // for(i = 0; i < n; ++i){
  //   if(group[i] == (j + 1)){
  //     sum_x[j] += x[i];
  //     sum_y[j] += y[i];
  //     nx[j]++;
  //     ny[j]++;
  //   }
  // }
  
  if(j < n){
    for(i = 0; i < 5; i++){
      if(group[j] == (i+1)){
        sum_x[i] += x[j];
        sum_y[i] += y[j];
        nx[i]++;
        ny[i]++;
      }
    }
  }
}

__global__ void recenter_step2(float *mu_x, float *mu_y, float *sum_x,
			       float *sum_y, int *nx, int *ny){
  int j = threadIdx.x;

  mu_x[j] = sum_x[j]/nx[j];
  mu_y[j] = sum_y[j]/ny[j];
}

void plot_data(const char *filename) {
  FILE *gnuplot = popen("gnuplot", "w");
  if (gnuplot == NULL) {
      perror("gnuplot 실행 실패");
      exit(EXIT_FAILURE);
  }

  // gnuplot 명령어
  // fprintf(gnuplot, "set terminal pngcairo size 1000,800 enhanced font 'Arial,14'\n");
  // fprintf(gnuplot, "set output '%s'\n", "test.png");
  fprintf(gnuplot, "set title 'Scatter Plot of Points'\n");
  fprintf(gnuplot, "set xlabel 'X Coordinate'\n");
  fprintf(gnuplot, "set ylabel 'Y Coordinate'\n");
  fprintf(gnuplot, "set grid\n");
  fprintf(gnuplot, "set term qt persist\n");  // 창 유지
  fprintf(gnuplot, "plot '%s' using 1:2:3 with points pointtype 7 pointsize 0.001 lc variable\n", filename);
  // fprintf(gnuplot, "set output\n");

  fflush(gnuplot);
  pclose(gnuplot);
}


void kmeans(int nreps, int n, int k,
            float *x_d, float *y_d, float *mu_x_d, float *mu_y_d,
            int *group_d, int *nx_d, int *ny_d,
            float *sum_x_d, float *sum_y_d, float *dst_d){}
  

void write_data_float(float *data, size_t n, char* f_name){
  int fd;
  int cnt = 0;
  int quo;
	size_t rem;
  ssize_t ret;
  size_t test_offset = 2147479552 / 4;

  fd = open(f_name, O_CREAT | O_RDWR, 0664);
  if(fd < 0)
    printf("File open Error\n");

  while(1){
    quo = n / 2147479552;
    // printf("quo: %d\n", quo);
    if(quo > 0){
      ret = write(fd, data+(cnt*test_offset), 2147479552);
      // printf("%zd\n", o_ret);
      cnt++;
      n -= 2147479552;
      if (ret == -1) { 
        printf("Error when writing\n");
        break;
      }			
    }
    else{
      rem = n % 2147479552;
      ret = write(fd, data+(cnt*test_offset), rem);
      printf("%zd\n", ret);
      break;
    }
  }

  close(fd);
}

void write_data_int(int *data, size_t n, char* f_name){
  int fd;
  int cnt = 0;
  int quo;
	size_t rem;
  ssize_t ret;
  size_t test_offset = 2147479552 / 4;

  fd = open(f_name, O_CREAT | O_RDWR, 0664);
  if(fd < 0)
    printf("File open Error\n");

  while(1){
    quo = n / 2147479552;
    // printf("quo: %d\n", quo);
    if(quo > 0){
      ret = write(fd, data+(cnt*test_offset), 2147479552);
      // printf("%zd\n", o_ret);
      cnt++;
      n -= 2147479552;
      if (ret == -1) { 
        printf("Error when writing\n");
        break;
      }			
    }
    else{
      rem = n % 2147479552;
      ret = write(fd, data+(cnt*test_offset), rem);
      printf("%zd\n", ret);
      break;
    }
  }

  close(fd);
}

void read_data_float(float *data, size_t n, char* f_name){
  int fd;
  int cnt = 0;
  int quo;
	size_t rem;
  ssize_t ret;
  size_t test_offset = 2147479552 / 4;

  fd = open(f_name, O_RDONLY);
  if(fd < 0)
    printf("File open Error\n");

  while(1){
    quo = n / 2147479552;
    // printf("quo: %d\n", quo);
    if(quo > 0){
      ret = read(fd, data+(cnt*test_offset), 2147479552);
      // printf("%zd\n", o_ret);
      cnt++;
      n -= 2147479552;
      if (ret == -1) { 
        printf("Error when writing\n");
        break;
      }			
    }
    else{
      rem = n % 2147479552;
      ret = read(fd, data+(cnt*test_offset), rem);
      printf("%zd\n", ret);
      break;
    }
  }

  close(fd);
}

void read_data_int(int *data, size_t n, char* f_name){
  int fd;
  int cnt = 0;
  int quo;
	size_t rem;
  ssize_t ret;
  size_t test_offset = 2147479552 / 4;

  fd = open(f_name, O_RDONLY);
  if(fd < 0)
    printf("File open Error\n");

  while(1){
    quo = n / 2147479552;
    // printf("quo: %d\n", quo);
    if(quo > 0){
      ret = read(fd, data+(cnt*test_offset), 2147479552);
      // printf("%zd\n", o_ret);
      cnt++;
      n -= 2147479552;
      if (ret == -1) { 
        printf("Error when writing\n");
        break;
      }			
    }
    else{
      rem = n % 2147479552;
      ret = read(fd, data+(cnt*test_offset), rem);
      printf("%zd\n", ret);
      break;
    }
  }

  close(fd);
}

void read_data(float **mu_x, float **mu_y, size_t *n, int *k,char* arg);
void print_results(int *group, float *mu_x, float *mu_y, int n, int k,char* argv);

int main(int argc,char* argv[]){
  
  int tid = atoi(argv[4]);
  int ret_dev_id;
  int mem_intensive = 1;

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

  /* cpu variables */
  // size_t n=300000000; /* number of points */
  size_t n=10000000; /* number of points */
  int k; /* number of clusters */
  // int *group;
  float *x = NULL, *y = NULL, *mu_x = NULL, *mu_y = NULL;

  /* gpu variables */
  int *group_d, *nx_d, *ny_d;
  float *x_d, *y_d, *mu_x_d, *mu_y_d, *sum_x_d, *sum_y_d, *dst_d;

  /* read data from files on cpu */
  read_data(&mu_x, &mu_y, &n, &k, argv[2]);

  dim3 grid(n, 1, 1);
  dim3 threads(k, 1, 1);

  size_t membytes = 0;
  membytes += page_align(n*sizeof(float));
  membytes += page_align(n*sizeof(float));
  membytes += page_align(n*sizeof(float));
  membytes += page_align(k*sizeof(float));
  membytes += page_align(k*sizeof(float));
  membytes += page_align(k*sizeof(float));
  membytes += page_align(k*sizeof(float));
  membytes += page_align(k*sizeof(int));
  membytes += page_align(k*sizeof(int));
  membytes += page_align(n*k*sizeof(float));
  membytes += 309 * 1024 * 1024;

  size_t max_k_usage = membytes;
	size_t k1_mem_req = 2 * page_align(n * sizeof(float)) + 2 * page_align(k * sizeof(float)) + page_align(n * k * sizeof(float));
  size_t k2_mem_req = page_align(n * sizeof(float)) + page_align(n * k * sizeof(float));
  size_t k3_mem_req = 4 * page_align(k * sizeof(float));
  size_t k4_mem_req = 3 * page_align(n * sizeof(float)) + 4 * page_align(k * sizeof(float));
  size_t k5_mem_req = 6 * page_align(k * sizeof(float));

  for(int loop = 0; loop < 5; loop++){
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
      
    //  long orig_alloc_mem = bemps_begin(tid, grid.x, grid.y, grid.z, threads.x, threads.y, threads.z, membytes, ret_dev_id);
    long orig_alloc_mem = bemps_begin(tid, grid.x, grid.y, grid.z, threads.x, threads.y, threads.z, max_k_usage, ret_dev_id, mem_intensive, ret_dev_id);
      
    clock_gettime( CLOCK_REALTIME, &specific_time);
    now = localtime(&specific_time.tv_sec);
    millsec = specific_time.tv_nsec;
      
    millsec = floor (specific_time.tv_nsec/1.0e6);
      
    printf("TID: %d after schedule, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
      now->tm_mon + 1, now->tm_mday, now->tm_hour, 
      now->tm_min, now->tm_sec, millsec);

    if (max_k_usage <= orig_alloc_mem){
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

    CUDA_CALL(cudaMallocManaged(&x_d, n*sizeof(float)));
    CUDA_CALL(cudaMallocManaged(&y_d, n*sizeof(float)));

    CUDA_CALL(cudaMallocManaged(&group_d,n*sizeof(int)));
    CUDA_CALL(cudaMallocManaged(&nx_d, k*sizeof(int)));
    CUDA_CALL(cudaMallocManaged(&ny_d, k*sizeof(int)));
    
    CUDA_CALL(cudaMallocManaged(&mu_x_d, k*sizeof(float)));
    CUDA_CALL(cudaMallocManaged(&mu_y_d, k*sizeof(float)));
    CUDA_CALL(cudaMallocManaged(&sum_x_d, k*sizeof(float)));
    CUDA_CALL(cudaMallocManaged(&sum_y_d, k*sizeof(float)));
    CUDA_CALL(cudaMallocManaged(&dst_d, n*k*sizeof(float)));

    auto t_stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(t_stop - t_start);
    std::cout << "Tid: " << tid << " Device mem allocation time: " << duration.count() << std::endl;
  
    t_start = high_resolution_clock::now();

    if(loop == 0){
      int fd = -1;
      ssize_t ret = -1;
      size_t test_offset = 2147479552 / 4;

      // fd = open("../../data/kmeans/300000000_points_x.txt", O_RDONLY);
      fd = open("../../data/kmeans/10000000points_x.txt", O_RDONLY);
      printf("fd: %d\n", fd);

      int cnt = 0;
      // size_t alloc_size = 1200000000;
      size_t alloc_size = 40000000;
      int quo;
      size_t rem;

      while(1){
        quo = alloc_size / 2147479552;
        printf("quo: %d\n", quo);
        if(quo > 0){
          ret = read(fd, x+(cnt*test_offset), 2147479552);
          printf("%zd\n", ret);
          cnt++;
          alloc_size -= 2147479552;	
        }
        else{
          rem = alloc_size % 2147479552;
          printf("rem: %zd\n", rem);
          ret = read(fd, x_d, rem);
          printf("ret: %zd\n", ret);
          break;
        }
      }
      close(fd);

      // fd = open("../../data/kmeans/300000000_points_y.txt", O_RDONLY);
      fd = open("../../data/kmeans/10000000points_y.txt", O_RDONLY);
      printf("fd: %d\n", fd);

      cnt = 0;
      alloc_size = 40000000;

      while(1){
        quo = alloc_size / 2147479552;
        printf("quo: %d\n", quo);
        if(quo > 0){
          ret = read(fd, y+(cnt*test_offset), 2147479552);
          printf("%zd\n", ret);
          cnt++;
          alloc_size -= 2147479552;		
        }
        else{
          rem = alloc_size % 2147479552;
          printf("rem: %zd\n", rem);
          ret = read(fd, y_d, rem);
          printf("%zd\n", ret);
          break;
        }
      }
      close(fd);

      printf("n: %d k: %d\n", n, k);

      memcpy(mu_x_d, mu_x, k*sizeof(float));
      memcpy(mu_y_d, mu_y, k*sizeof(float));
    }
    else{
      read_data_float(x_d, n * sizeof(float), "x_d.txt");
      read_data_float(y_d, n * sizeof(float), "y_d.txt");
      read_data_int(group_d, n * sizeof(int), "group_d.txt");
      read_data_int(nx_d, k * sizeof(int), "nx_d.txt");
      read_data_int(ny_d, k * sizeof(int), "ny_d.txt");
      read_data_float(mu_x_d, k * sizeof(float), "mu_x_d.txt");
      read_data_float(mu_y_d, k * sizeof(float), "mu_y_d.txt");
      read_data_float(sum_x_d, k * sizeof(float), "sum_x_d.txt");
      read_data_float(sum_y_d, k * sizeof(float), "sum_y_d.txt");
      read_data_float(dst_d, n * k * sizeof(float), "dst_d.txt");
    }

    t_stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(t_stop - t_start);
    std::cout << "Tid: " << tid << " Storage to Host memcpy time: " << duration.count() << std::endl;

    ret1.devPtr = x_d;
    ret1.advice = cudaMemAdviseSetPreferredLocation;
    ret1.device = ret_dev_id;
    ret1.alloc_size = n*sizeof(float);
  
    ret2.devPtr = y_d;
    ret2.advice = cudaMemAdviseSetPreferredLocation;
    ret2.device = ret_dev_id;
    ret2.alloc_size = n*sizeof(float);
  
    ret3.devPtr = group_d;
    ret3.advice = cudaMemAdviseSetPreferredLocation;
    ret3.device = ret_dev_id;
    ret3.alloc_size = n*sizeof(float);
  
    ret4.devPtr = nx_d;
    ret4.advice = cudaMemAdviseSetPreferredLocation;
    ret4.device = ret_dev_id;
    ret4.alloc_size = k*sizeof(int);

    ret5.devPtr = ny_d;
    ret5.advice = cudaMemAdviseSetPreferredLocation;
    ret5.device = ret_dev_id;
    ret5.alloc_size = k*sizeof(int);

    ret6.devPtr = mu_x_d;
    ret6.advice = cudaMemAdviseSetPreferredLocation;
    ret6.device = ret_dev_id;
    ret6.alloc_size = k*sizeof(float);
  
    ret7.devPtr = mu_y_d;
    ret7.advice = cudaMemAdviseSetPreferredLocation;
    ret7.device = ret_dev_id;
    ret7.alloc_size = k*sizeof(float);
  
    ret8.devPtr = sum_x_d;
    ret8.advice = cudaMemAdviseSetPreferredLocation;
    ret8.device = ret_dev_id;
    ret8.alloc_size = k*sizeof(float);
  
    ret9.devPtr = sum_y_d;
    ret9.advice = cudaMemAdviseSetPreferredLocation;
    ret9.device = ret_dev_id;
    ret9.alloc_size = k*sizeof(int);

    ret10.devPtr = dst_d;
    ret10.advice = cudaMemAdviseSetPreferredLocation;
    ret10.device = ret_dev_id;
    ret10.alloc_size = n*k*sizeof(float);

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

    t_start = high_resolution_clock::now();
  
    if(full == 1){
      for(Parameters ret : mem_list){
        CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
        CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, ret.device, 0));
      }
    }
  
    t_stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(t_stop - t_start);
    std::cout << "Tid: " << tid << " Host to Device memcpy time: " << duration.count() << std::endl;

    int i;
    
    int nreps = 10;
    int ready_to_launch = 0;
    // size_t os_110_dev = 1572864000;
    size_t os_110_dev = 786432000;

    float total_e;

    int tmp_g, tmp_b;
    tmp_g = (n / 1024) + 1;
    tmp_b = 1024;

    cudaEvent_t *k1_start, *k1_stop, *k2_start, *k2_stop, *k3_start, *k3_stop, *k4_start, *k4_stop, *k5_start, *k5_stop;

    k1_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 10);
    k1_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 10);
    k2_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 10);
    k2_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 10);
    k3_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 10);
    k3_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 10);
    k4_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 10);
    k4_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 10);
    k5_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 10);
    k5_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 10);

    cudaStream_t s1, s2, s3, s4, s5;
    cudaEvent_t event1, event2, event3, event4, event5;

    int task_cnt = 0;

    for(i = 0; i < nreps; ++i){
      el_wait(tid, ready_to_launch);
      
      if((full == 0) && (!ready_to_launch)){
        el_wait(tid, ready_to_launch);
        u_mem = bemps_extra_task_mem(tid);
        
        while((u_mem + os_110_dev) < k1_mem_req){
          u_mem = bemps_extra_task_mem(tid);
          el_wait(tid, ready_to_launch);

          if(((u_mem + os_110_dev) >= k1_mem_req) || (ready_to_launch))
            break;
        }
        if(u_mem == max_k_usage){
          full = 1;
          priority = -5;
        }
        // checking 
      }

      CUDA_CHECK(cudaStreamCreateWithPriority(&s1, 0, priority));
      // CUDA_CHECK(cudaStreamCreate(&s1));

      CUDA_CHECK(cudaEventCreateWithFlags(&event1, cudaEventDisableTiming));

      CUDA_CHECK(cudaEventCreate(&k1_start[i]));
      CUDA_CHECK(cudaEventCreateWithFlags(&k1_stop[i], cudaEventBlockingSync));
      CUDA_CHECK(cudaEventRecord(k1_start[i], s1));

      get_dst<<<n, k, 0, s1>>>(dst_d, x_d, y_d, mu_x_d, mu_y_d);
      
      CUDA_CHECK(cudaEventRecord(event1, s1));

      task_monitoring(event1, tid, orig_alloc_mem, membytes);
                  
      if(i == 0)
        launch_signal(tid);
      
      CUDA_CHECK(cudaEventRecord(k1_stop[i], s1));

      CUDA_CHECK(cudaEventSynchronize(k1_stop[i]));

      CUDA_CHECK(cudaStreamDestroy(s1));
      CUDA_CHECK(cudaEventDestroy(event1));

      el_wait(tid, ready_to_launch);
      
      if((full == 0) && (!ready_to_launch)){
        el_wait(tid, ready_to_launch);
        u_mem = bemps_extra_task_mem(tid);
        
        while((u_mem + os_110_dev) < k2_mem_req){
          u_mem = bemps_extra_task_mem(tid);
          el_wait(tid, ready_to_launch);

          if(((u_mem + os_110_dev) >= k2_mem_req) || (ready_to_launch))
            break;
        }
        if(u_mem == max_k_usage){
          full = 1;
          priority = -5;
        }
        // checking 
      }

      CUDA_CHECK(cudaStreamCreateWithPriority(&s2, 0, priority));
      // CUDA_CHECK(cudaStreamCreate(&s2));

      CUDA_CHECK(cudaEventCreateWithFlags(&event2, cudaEventDisableTiming));

      CUDA_CHECK(cudaEventCreate(&k2_start[i]));
      CUDA_CHECK(cudaEventCreateWithFlags(&k2_stop[i], cudaEventBlockingSync));
      CUDA_CHECK(cudaEventRecord(k2_start[i], s2));

      regroup<<<n, 1, 0, s2>>>(group_d, dst_d, k);

      CUDA_CHECK(cudaEventRecord(event2, s2));

      task_monitoring(event2, tid, orig_alloc_mem, membytes);
                  
      CUDA_CHECK(cudaEventRecord(k2_stop[i], s2));

      CUDA_CHECK(cudaEventSynchronize(k2_stop[i]));

      CUDA_CHECK(cudaStreamDestroy(s2));
      CUDA_CHECK(cudaEventDestroy(event2));

      el_wait(tid, ready_to_launch);
      
      if((full == 0) && (!ready_to_launch)){
        el_wait(tid, ready_to_launch);
        u_mem = bemps_extra_task_mem(tid);
        
        while((u_mem + os_110_dev) < k3_mem_req){
          u_mem = bemps_extra_task_mem(tid);
          el_wait(tid, ready_to_launch);

          if(((u_mem + os_110_dev) >= k3_mem_req) || (ready_to_launch))
            break;
        }
        if(u_mem == max_k_usage){
          full = 1;
          priority = -5;
        }
        // checking 
      }

      CUDA_CHECK(cudaStreamCreateWithPriority(&s3, 0, priority));
      // CUDA_CHECK(cudaStreamCreate(&s3));

      CUDA_CHECK(cudaEventCreateWithFlags(&event3, cudaEventDisableTiming));

      CUDA_CHECK(cudaEventCreate(&k3_start[i]));
      CUDA_CHECK(cudaEventCreateWithFlags(&k3_stop[i], cudaEventBlockingSync));
      CUDA_CHECK(cudaEventRecord(k3_start[i], s3));

      clear<<<1, k, 0, s3>>>(sum_x_d, sum_y_d, nx_d, ny_d);

      CUDA_CHECK(cudaEventRecord(event3, s3));

      task_monitoring(event3, tid, orig_alloc_mem, membytes);
                  
      CUDA_CHECK(cudaEventRecord(k3_stop[i], s3));

      CUDA_CHECK(cudaEventSynchronize(k3_stop[i]));

      CUDA_CHECK(cudaStreamDestroy(s3));
      CUDA_CHECK(cudaEventDestroy(event3));

      el_wait(tid, ready_to_launch);
      
      if((full == 0) && (!ready_to_launch)){
        el_wait(tid, ready_to_launch);
        u_mem = bemps_extra_task_mem(tid);
        
        while((u_mem + os_110_dev) < k4_mem_req){
          u_mem = bemps_extra_task_mem(tid);
          el_wait(tid, ready_to_launch);

          if(((u_mem + os_110_dev) >= k4_mem_req) || (ready_to_launch))
            break;
        }
        if(u_mem == max_k_usage){
          full = 1;
          priority = -5;
        }
        // checking 
      }

      CUDA_CHECK(cudaStreamCreateWithPriority(&s4, 0, priority));
      // CUDA_CHECK(cudaStreamCreate(&s4));

      CUDA_CHECK(cudaEventCreateWithFlags(&event4, cudaEventDisableTiming));

      CUDA_CHECK(cudaEventCreate(&k4_start[i]));
      CUDA_CHECK(cudaEventCreateWithFlags(&k4_stop[i], cudaEventBlockingSync));
      CUDA_CHECK(cudaEventRecord(k4_start[i], s4));

      // recenter_step1<<<1,k>>>(sum_x_d, sum_y_d, nx_d, ny_d, x_d, y_d, group_d, n);
      recenter_step1<<<tmp_g, tmp_b, 0, s4>>>(sum_x_d, sum_y_d, nx_d, ny_d, x_d, y_d, group_d, n);
      
      CUDA_CHECK(cudaEventRecord(event4, s4));

      task_monitoring(event4, tid, orig_alloc_mem, membytes);
                  
      CUDA_CHECK(cudaEventRecord(k4_stop[i], s4));

      CUDA_CHECK(cudaEventSynchronize(k4_stop[i]));

      CUDA_CHECK(cudaStreamDestroy(s4));
      CUDA_CHECK(cudaEventDestroy(event4));

      el_wait(tid, ready_to_launch);
      
      if((full == 0) && (!ready_to_launch)){
        el_wait(tid, ready_to_launch);
        u_mem = bemps_extra_task_mem(tid);
        
        while((u_mem + os_110_dev) < k5_mem_req){
          u_mem = bemps_extra_task_mem(tid);
          el_wait(tid, ready_to_launch);

          if(((u_mem + os_110_dev) >= k5_mem_req) || (ready_to_launch))
            break;
        }
        if(u_mem == max_k_usage){
          full = 1;
          priority = -5;
        }
        // checking 
      }

      CUDA_CHECK(cudaStreamCreateWithPriority(&s5, 0, priority));
      // CUDA_CHECK(cudaStreamCreate(&s5));

      CUDA_CHECK(cudaEventCreateWithFlags(&event5, cudaEventDisableTiming));

      CUDA_CHECK(cudaEventCreate(&k5_start[i]));
      CUDA_CHECK(cudaEventCreateWithFlags(&k5_stop[i], cudaEventBlockingSync));
      CUDA_CHECK(cudaEventRecord(k5_start[i], s5));

      recenter_step2<<<1, k, 0, s5>>>(mu_x_d, mu_y_d, sum_x_d, sum_y_d, nx_d, ny_d);
      
      CUDA_CHECK(cudaEventRecord(event5, s5));

      task_monitoring(event5, tid, orig_alloc_mem, membytes);
                  
      CUDA_CHECK(cudaEventRecord(k5_stop[i], s5));

      CUDA_CHECK(cudaEventSynchronize(k5_stop[i]));

      CUDA_CHECK(cudaStreamDestroy(s5));
      CUDA_CHECK(cudaEventDestroy(event5));
    }

    float total_k1 = 0;
    float total_k2 = 0;
    float total_k3 = 0;
    float total_k4 = 0;
    float total_k5 = 0;
    float time = 0;

    for(i = 0; i < 10; i++){
      cudaEventElapsedTime(&time, k1_start[i], k2_stop[i]);
      // printf("%f ", time);
      total_k1 += time;
      
      cudaEventElapsedTime(&time, k2_start[i], k2_stop[i]);
      // printf("%f ", time);
      total_k2 += time;
      
      cudaEventElapsedTime(&time, k3_start[i], k3_stop[i]);
      // printf("%f\n", time);
      total_k3 += time;

      cudaEventElapsedTime(&time, k4_start[i], k4_stop[i]);
      // printf("%f ", time);
      total_k4 += time;
      
      cudaEventElapsedTime(&time, k5_start[i], k5_stop[i]);
      // printf("%f ", time);
      total_k5 += time;
    }
    
    printf("K1_kernel: %f, K2_kernel: %f, K3_kernel: %f, K4_kernel: %f, K5_kernel: %f\n", total_k1, total_k2, total_k3, total_k4, total_k5);

    t_start = high_resolution_clock::now();
  
    for(Parameters ret : mem_list){
      CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, cudaCpuDeviceId, 0));
    }
  
    t_stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(t_stop - t_start);
    std::cout << "Tid: " << tid << " Device to Host memcpy time: " << duration.count() << std::endl;

    t_start = high_resolution_clock::now();

    char str1[100];
		sprintf(str1, "%d", tid);
    char *filename = "points.dat";

    strcat(str1, filename);

    FILE *file = fopen(str1, "w");

    if (file == NULL) {
      printf("File open error\n");
    }

    for (int j = 0; j < n; j++) {
      fprintf(file, "%f %f %d\n", x_d[j], y_d[j], group_d[j]);
    }

    fclose(file);

    write_data_float(x_d, n * sizeof(float), "x_d.txt");
    write_data_float(y_d, n * sizeof(float), "y_d.txt");
    write_data_int(group_d, n * sizeof(int), "group_d.txt");
    write_data_int(nx_d, k * sizeof(int), "nx_d.txt");
    write_data_int(ny_d, k * sizeof(int), "ny_d.txt");
    write_data_float(mu_x_d, k * sizeof(float), "mu_x_d.txt");
    write_data_float(mu_y_d, k * sizeof(float), "mu_y_d.txt");
    write_data_float(sum_x_d, k * sizeof(float), "sum_x_d.txt");
    write_data_float(sum_y_d, k * sizeof(float), "sum_y_d.txt");
    write_data_float(dst_d, n * k * sizeof(float), "dst_d.txt");

    t_stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(t_stop - t_start);
    std::cout << "Tid: " << tid << " Host to Storage memcpy time: " << duration.count() << std::endl;

    free(k1_start);
    free(k1_stop);
    free(k2_start);
    free(k2_stop);
    free(k3_start);
    free(k3_stop);
    free(k4_start);
    free(k4_stop);
    free(k5_start);
    free(k5_stop);
    // free(mu_x);
    // free(mu_y);
    // free(group);

    t_start = high_resolution_clock::now();

    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
    CUDA_CHECK(cudaFree(mu_x_d));
    CUDA_CHECK(cudaFree(mu_y_d));
    CUDA_CHECK(cudaFree(group_d));
    CUDA_CHECK(cudaFree(nx_d));
    CUDA_CHECK(cudaFree(ny_d));
    CUDA_CHECK(cudaFree(sum_x_d));
    CUDA_CHECK(cudaFree(sum_y_d));
    CUDA_CHECK(cudaFree(dst_d));
    CUDA_CHECK(cudaDeviceReset());

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

    
    // plot_data(str1);

    std::vector<Parameters> tmp_list;
    tmp_list.swap(mem_list);
    tid += 9;
  }

  return 0;
}

void read_data(float **mu_x, float **mu_y, size_t *n, int *k,char* arg){
  FILE *fp;
  char buf[64];

  // *n = 300000000;
  *n = 10000000;

  *k = 0;
  fp = fopen("../../data/kmeans/initCoord.txt", "r");
  while(fgets(buf, 64, fp) != NULL){
    *k += 1;
    *mu_x = (float*) realloc(*mu_x, (*k)*sizeof(float));
    *mu_y = (float*) realloc(*mu_y, (*k)*sizeof(float));
    std::istringstream line_stream(buf);
    float x1,y1;
    line_stream >> x1 >> y1;
    (*mu_x)[*k - 1] = x1;
    (*mu_y)[*k - 1] = x1;
  }
  fclose(fp);
}


void print_results(int *group, float *mu_x, float *mu_y, int n, int k,char* arg){
  printf("HI1\n");
  FILE *fp;
  int i;
  std::string str(arg),str1,str2;
  str = "result/cuda/" + str;

  printf("HI2\n");

   str1 = str + "_group_members.txt";
  // fp = fopen(str1.c_str(), "w");
  fp = fopen("group_members.txt", "w");
  for(i = 0; i < n; ++i){
    fprintf(fp, "%d\n", group[i]);
  }
  fclose(fp);
  
  printf("HI3\n");

  str2 = str + "_centroids.txt";
  // fp = fopen(str2.c_str(), "w");
  fp = fopen("centroids.txt", "w");
  for(i = 0; i < k; ++i){
    fprintf(fp, "%0.6f %0.6f\n", mu_x[i], mu_y[i]);
  }
  fclose(fp);

  printf("HI4\n");

  fp = fopen("CUDAtimes.txt", "a");
    fprintf(fp, "%0.6f\n", gpu_time_used);

    printf("HI5\n");

fclose(fp);
}
