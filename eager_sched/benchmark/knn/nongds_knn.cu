/** Modifed version of knn-CUDA from https://github.com/vincentfpgarcia/kNN-CUDA
 * The modifications are
 *      removed texture memory usage
 *      removed split query KNN computation
 *      added feature extraction with bilinear interpolation
 *
 * Last modified by Christopher B. Choy <chrischoy@ai.stanford.edu> 12/23/2016
 */

// Includes
#include "cuda.h"
#include <cstdio>
#include <sys/time.h>
#include <time.h>

#include <bemps.hpp>
#include <fcntl.h>
#include <assert.h>
#include <unistd.h>
#include <algorithm>
#include <vector>
#include <chrono>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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

// Constants used by the program
#define BLOCK_DIM 16

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

//-----------------------------------------------------------------------------------------------//
//                                            KERNELS //
//-----------------------------------------------------------------------------------------------//
__global__ void extract_with_interpolation(int nthreads, float *data,
                                           float *n_xy_coords,
                                           float *extracted_data,
                                           int n_max_coord, int channels,
                                           int height, int width) {

  int x0, x1, y0, y1, nc;
  float wx0, wx1, wy0, wy1;
  int n, nd;
  float x, y;

  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (nthreads);
       index += blockDim.x * gridDim.x) {
    n = (index / n_max_coord);
    nd = n * n_max_coord * channels;
    x = n_xy_coords[index * 2];
    y = n_xy_coords[index * 2 + 1];

    x0 = static_cast<int>(floor(x));
    x1 = x0 + 1;
    y0 = static_cast<int>(floor(y));
    y1 = y0 + 1;

    x0 = x0 <= 0 ? 0 : (x0 >= (width - 1) ? (width - 1) : x0);
    y0 = y0 <= 0 ? 0 : (y0 >= (height - 1) ? (height - 1) : y0);
    x1 = x1 <= 0 ? 0 : (x1 >= (width - 1) ? (width - 1) : x1);
    y1 = y1 <= 0 ? 0 : (y1 >= (height - 1) ? (height - 1) : y1);

    wx0 = static_cast<float>(x1) - x;
    wx1 = x - x0;
    wy0 = static_cast<float>(y1) - y;
    wy1 = y - y0;

    if (x0 == x1) {
      wx0 = 1;
      wx1 = 0;
    }
    if (y0 == y1) {
      wy0 = 1;
      wy1 = 0;
    }
    for (int c = 0; c < channels; c++) {
      nc = (n * channels + c) * height;
      // extracted_data[index * channels + c] = wy0 * wx0 * data[(nc + y0) *
      // width + x0]
      // extracted_data[nd + index % n_max_coord + n_max_coord * c] = index;
      extracted_data[nd + index % n_max_coord + n_max_coord * c] =
          wy0 * wx0 * data[(nc + y0) * width + x0] +
          wy1 * wx0 * data[(nc + y1) * width + x0] +
          wy0 * wx1 * data[(nc + y0) * width + x1] +
          wy1 * wx1 * data[(nc + y1) * width + x1];
    }
  }
}

/**
  * Computes the distance between two matrix A (reference points) and
  * B (query points) containing respectively wA and wB points.
  *
  * @param A     pointer on the matrix A
  * @param wA    width of the matrix A = number of points in A
  * @param B     pointer on the matrix B
  * @param wB    width of the matrix B = number of points in B
  * @param dim   dimension of points = height of matrices A and B
  * @param AB    pointer on the matrix containing the wA*wB distances computed
  */
__global__ void cuComputeDistanceGlobal(float *A, int wA, float *B, int wB,
                                        int dim, float *AB) {

  // Declaration of the shared memory arrays As and Bs used to store the
  // sub-matrix of A and B
  __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
  __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

  // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
  __shared__ int begin_A;
  __shared__ int begin_B;
  __shared__ int step_A;
  __shared__ int step_B;
  __shared__ int end_A;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Other variables
  float tmp;
  float ssd = 0;

  // Loop parameters
  begin_A = BLOCK_DIM * blockIdx.y;
  begin_B = BLOCK_DIM * blockIdx.x;
  step_A = BLOCK_DIM * wA;
  step_B = BLOCK_DIM * wB;
  end_A = begin_A + (dim - 1) * wA;

  // Conditions
  int cond0 = (begin_A + tx < wA); // used to write in shared memory
  int cond1 = (begin_B + tx < wB); // used to write in shared memory & to
                                   // computations and to write in output matrix
  int cond2 =
      (begin_A + ty < wA); // used to computations and to write in output matrix
  // Loop over all the sub-matrices of A and B required to compute the block
  // sub-matrix
  for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {
    // Load the matrices from device memory to shared memory; each thread loads
    // one element of each matrix
    if (a / wA + ty < dim) {
      shared_A[ty][tx] = (cond0) ? A[a + wA * ty + tx] : 0;
      shared_B[ty][tx] = (cond1) ? B[b + wB * ty + tx] : 0;
    } else {
      shared_A[ty][tx] = 0;
      shared_B[ty][tx] = 0;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();
    // Compute the difference between the two matrixes; each thread computes one
    // element of the block sub-matrix
    if (cond2 && cond1) {
      for (int k = 0; k < BLOCK_DIM; ++k) {
        tmp = shared_A[k][ty] - shared_B[k][tx];
        ssd += tmp * tmp;
      }
    }

    // Synchronize to make sure that the preceding computation is done before
    // loading two new sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory; each thread writes one element
  if (cond2 && cond1) {
    AB[(begin_A + ty) * wB + begin_B + tx] = ssd;
  }
}

/**
  * Gathers k-th smallest distances for each column of the distance matrix in
 * the top.
  *
  * @param dist        distance matrix
  * @param ind         index matrix
  * @param width       width of the distance matrix and of the index matrix
  * @param height      height of the distance matrix and of the index matrix
  * @param k           number of neighbors to consider
  */
__global__ void cuInsertionSort(float *dist, int *ind, int width, int height,
                                int k) {
  // printf("test2\n");
  // Variables
  int l, i, j;
  float *p_dist;
  int *p_ind;
  float curr_dist, max_dist;
  int curr_row, max_row;
  unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if (xIndex < width) {
    // Pointer shift, initialization, and max value
    p_dist = dist + xIndex;
    p_ind = ind + xIndex;
    max_dist = p_dist[0];
    p_ind[0] = 0;

    // Part 1 : sort kth firt elementZ
    for (l = 1; l < k; l++) {
      curr_row = l * width;
      curr_dist = p_dist[curr_row];
      if (curr_dist < max_dist) {
        i = l - 1;
        for (int a = 0; a < l - 1; a++) {
          if (p_dist[a * width] > curr_dist) {
            i = a;
            break;
          }
        }
        for (j = l; j > i; j--) {
          p_dist[j * width] = p_dist[(j - 1) * width];
          p_ind[j * width] = p_ind[(j - 1) * width];
        }
        p_dist[i * width] = curr_dist;
        p_ind[i * width] = l;
      } else {
        p_ind[l * width] = l;
      }
      max_dist = p_dist[curr_row];
    }

    // Part 2 : insert element in the k-th first lines
    max_row = (k - 1) * width;
    for (l = k; l < height; l++) {
      curr_dist = p_dist[l * width];
      if (curr_dist < max_dist) {
        i = k - 1;
        for (int a = 0; a < k - 1; a++) {
          if (p_dist[a * width] > curr_dist) {
            i = a;
            break;
          }
        }
        for (j = k - 1; j > i; j--) {
          p_dist[j * width] = p_dist[(j - 1) * width];
          p_ind[j * width] = p_ind[(j - 1) * width];
        }
        p_dist[i * width] = curr_dist;
        p_ind[i * width] = l;
        max_dist = p_dist[max_row];
      }
    }
  }
}

/**
  * Computes the square root of the first line (width-th first element)
  * of the distance matrix.
  *
  * @param dist    distance matrix
  * @param width   width of the distance matrix
  * @param k       number of neighbors to consider
  */
__global__ void cuParallelSqrt(float *dist, int width, int k) {
  unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  // printf("test3\n");
  if (xIndex < width && yIndex < k)
    dist[yIndex * width + xIndex] = sqrt(dist[yIndex * width + xIndex]);
}

//-----------------------------------------------------------------------------------------------//
//                                   K-th NEAREST NEIGHBORS //
//-----------------------------------------------------------------------------------------------//

/**
  * Prints the error message return during the memory allocation.
  *
  * @param error        error value return by the memory allocation function
  * @param memorySize   size of memory tried to be allocated
  */
void printErrorMessage(cudaError_t error, int memorySize) {
  printf("==================================================\n");
  printf("MEMORY ALLOCATION ERROR  : %s\n", cudaGetErrorString(error));
  printf("Whished allocated memory : %d\n", memorySize);
  printf("==================================================\n");
}

/**
  * K nearest neighbor algorithm
  * - Initialize CUDA
  * - Allocate device memory
  * - Copy point sets (reference and query points) from host to device memory
  * - Compute the distances + indexes to the k nearest neighbors for each query
 * point
  * - Copy distances from device to host memory
  *
  * @param ref_host      reference points ; pointer to linear matrix
  * @param ref_width     number of reference points ; width of the matrix
  * @param query_host    query points ; pointer to linear matrix
  * @param query_width   number of query points ; width of the matrix
  * @param height        dimension of points ; height of the matrices
  * @param k             number of neighbor to consider
  * @param dist_host     distances to k nearest neighbors ; pointer to linear
 * matrix
  * @param dist_host     indexes of the k nearest neighbors ; pointer to linear
 * matrix
  *
  */
void knn_cuda(float *ref_host, int ref_width, float *query_host,
              int query_width, int height, int k, float *dist_host,
              int *ind_host) {

  // Variables

  // CUDA Initialisation
//   cuInit(0);

  // Grids ans threads
  dim3 g_16x16(query_width / 16, ref_width / 16, 1);
  dim3 t_16x16(16, 16, 1);
  if (query_width % 16 != 0)
    g_16x16.x += 1;
  if (ref_width % 16 != 0)
    g_16x16.y += 1;
  //
  dim3 g_256x1(query_width / 256, 1, 1);
  dim3 t_256x1(256, 1, 1);
  if (query_width % 256 != 0)
    g_256x1.x += 1;

  dim3 g_k_16x16(query_width / 16, k / 16, 1);
  dim3 t_k_16x16(16, 16, 1);
  if (query_width % 16 != 0)
    g_k_16x16.x += 1;
  if (k % 16 != 0)
    g_k_16x16.y += 1;

  
    // printf("%d %d %d %d %d\n", g_16x16.x, g_16x16.y,  g_256x1.x, g_k_16x16.x, g_k_16x16.y);
  // Kernel 1: Compute all the distances
  cuComputeDistanceGlobal<<<g_16x16, t_16x16>>>(ref_host, ref_width, query_host,
                                                query_width, height, dist_host);
  // Kernel 2: Sort each column
  cuInsertionSort<<<g_256x1, t_256x1>>>(dist_host, ind_host, query_width,
                                        ref_width, k);
  // Kernel 3: Compute square root of k first elements
  cuParallelSqrt<<<g_k_16x16, t_k_16x16>>>(dist_host, query_width, k);
  cudaDeviceSynchronize();
}

/**
  * Example of use of kNN search CUDA.
  */
int main(int argc, char** argv) {
  // Variables and parameters
  float *ref;           // Pointer to reference point array
  float *query;         // Pointer to query point array
  float *dist, *dist_c; // Pointer to distance array
  int *ind, *ind_c;     // Pointer to index array
  int ref_nb = 32768;    // Reference point number, max=65535
  int query_nb = 32768;  // Query point number,     max=65535
  int dim = 32;         // Dimension of points
  int k = 20;           // Nearest neighbors to consider
  int iterations = 100;
  int c_iterations = 10;
  int i;
  const float precision = 0.001f; // distance error max
  int nb_correct_precisions = 0;
  int nb_correct_indexes = 0;
  // float *knn_dist = (float *)malloc(query_nb * k * sizeof(float));
  // int *knn_index = (int *)malloc(query_nb * k * sizeof(int));

  int tid = atoi(argv[1]);
  int ret_dev_id;
  int mem_intensive = 1;

  Parameters ret1;
	Parameters ret2;
	Parameters ret3;
	Parameters ret4;

  dim3  grid(2048, 1, 1);
	dim3  threads(256, 1, 1);

  dim3 g_16x16(query_nb / 16, ref_nb / 16, 1);
  dim3 t_16x16(16, 16, 1);
  if (query_nb % 16 != 0)
    g_16x16.x += 1;
  if (ref_nb % 16 != 0)
    g_16x16.y += 1;
  //
  dim3 g_256x1(query_nb / 256, 1, 1);
  dim3 t_256x1(256, 1, 1);
  if (query_nb % 256 != 0)
    g_256x1.x += 1;

  dim3 g_k_16x16(query_nb / 16, k / 16, 1);
  dim3 t_k_16x16(16, 16, 1);
  if (query_nb % 16 != 0)
    g_k_16x16.x += 1;
  if (k % 16 != 0)
    g_k_16x16.y += 1;

  size_t membytes = 0;
  membytes += page_align(ref_nb * dim * sizeof(float));
  membytes += page_align(query_nb * dim * sizeof(float));
  membytes += page_align(query_nb * ref_nb * sizeof(float));
  membytes += page_align(query_nb * k * sizeof(int));
  membytes += 309 * 1024 * 1024;

  size_t max_k_usage = membytes;
	size_t k1_mem_req = page_align(ref_nb * dim * sizeof(float)) + page_align(query_nb * dim * sizeof(float)) + page_align(query_nb * ref_nb * sizeof(float));
  size_t k2_mem_req = page_align(query_nb * ref_nb * sizeof(float)) + page_align(query_nb * k * sizeof(int));
  size_t k3_mem_req = page_align(query_nb * ref_nb * sizeof(float));

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
  // Memory allocation
  // ref = (float *)malloc(ref_nb * dim * sizeof(float));
  // query = (float *)malloc(query_nb * dim * sizeof(float));
  cudaMallocManaged(&ref, ref_nb * dim * sizeof(float));
  cudaMallocManaged(&query, query_nb * dim * sizeof(float));
  cudaMallocManaged(&dist, query_nb * ref_nb * sizeof(float));
  cudaMallocManaged(&ind, query_nb * k * sizeof(int));

  auto t_stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Device mem allocation time: " << duration.count() << std::endl;
 
	t_start = high_resolution_clock::now();

  // dist_c = (float *)malloc(query_nb * k * sizeof(float));
  // ind_c = (int *)malloc(query_nb * k * sizeof(float));

  // Init
  srand(time(NULL));
  for (i = 0; i < ref_nb * dim; i++)
    ref[i] = (float)rand() / (float)RAND_MAX;
  for (i = 0; i < query_nb * dim; i++)
    query[i] = (float)rand() / (float)RAND_MAX;

  t_stop = high_resolution_clock::now();
  duration = duration_cast<milliseconds>(t_stop - t_start);
  std::cout << "Tid: " << tid << " Storage to Host memcpy time: " << duration.count() << std::endl;

  ret1.devPtr = ref;
	ret1.advice = cudaMemAdviseSetPreferredLocation;
	ret1.device = ret_dev_id;
	ret1.alloc_size = ref_nb * dim * sizeof(float);
 
	ret2.devPtr = query;
	ret2.advice = cudaMemAdviseSetPreferredLocation;
	ret2.device = ret_dev_id;
	ret2.alloc_size = query_nb * dim * sizeof(float);
 
	ret3.devPtr = dist;
	ret3.advice = cudaMemAdviseSetPreferredLocation;
	ret3.device = ret_dev_id;
	ret3.alloc_size = query_nb * ref_nb * sizeof(float);
 
	ret4.devPtr = ind;
	ret4.advice = cudaMemAdviseSetPreferredLocation;
	ret4.device = ret_dev_id;
	ret4.alloc_size = query_nb * k * sizeof(int);

  mem_list.push_back(ret1);
	mem_list.push_back(ret2);
	mem_list.push_back(ret3);
	mem_list.push_back(ret4);
  
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

  // Display informations
  printf("Number of reference points      : %6d\n", ref_nb);
  printf("Number of query points          : %6d\n", query_nb);
  printf("Dimension of points             : %4d\n", dim);
  printf("Number of neighbors to consider : %4d\n", k);
  printf("Processing kNN search           :\n");

  float precision_accuracy;
  float index_accuracy;

  cudaEvent_t *k1_start, *k1_stop, *k2_start, *k2_stop, *k3_start, *k3_stop;

  k1_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 100);
	k1_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 100);
	k2_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 100);
	k2_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 100);
	k3_start = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 100);
	k3_stop = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * 100);

  cudaStream_t s1, s2, s3;
	cudaEvent_t event1, event2, event3;

  int ready_to_launch = 0;
	// size_t os_110_dev = 1572864000;
	size_t os_110_dev = 786432000;
	// size_t os_110_dev = 3145728000;

  // Call kNN search CUDA
  // CUDA_CHECK(cudaEventRecord(start, 0));
  for (i = 0; i < iterations; i++) {
    
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

    // Kernel 1: Compute all the distances
    cuComputeDistanceGlobal<<<g_16x16, t_16x16, 0, s1>>>(ref, ref_nb, query, query_nb, dim, dist);
    
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

    // Kernel 2: Sort each column
    cuInsertionSort<<<g_256x1, t_256x1, 0, s2>>>(dist, ind, query_nb, ref_nb, k);
    
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

    // Kernel 3: Compute square root of k first elements
    cuParallelSqrt<<<g_k_16x16, t_k_16x16, 0, s3>>>(dist, query_nb, k);

    CUDA_CHECK(cudaEventRecord(event3, s3));

		task_monitoring(event3, tid, orig_alloc_mem, membytes);
								
		CUDA_CHECK(cudaEventRecord(k3_stop[i], s3));

		CUDA_CHECK(cudaEventSynchronize(k3_stop[i]));

    CUDA_CHECK(cudaStreamDestroy(s3));
		CUDA_CHECK(cudaEventDestroy(event3));
  }

  float total_k1 = 0;
  float total_k2 = 0;
  float total_k3 = 0;
  float time = 0;

  for(i = 0; i < 100; i++){
		cudaEventElapsedTime(&time, k1_start[i], k2_stop[i]);
		// printf("%f ", time);
		total_k1 += time;
		
		cudaEventElapsedTime(&time, k2_start[i], k2_stop[i]);
		// printf("%f ", time);
		total_k2 += time;
		
    cudaEventElapsedTime(&time, k3_start[i], k3_stop[i]);
		// printf("%f\n", time);
		total_k3 += time;
	}
  
  printf("K1_kernel: %f, K2_kernel: %f, K3_kernel: %f\n", total_k1, total_k2, total_k3);

  t_start = high_resolution_clock::now();

	CUDA_CHECK(cudaMemPrefetchAsync(dist, query_nb * ref_nb * sizeof(float), cudaCpuDeviceId, 0));

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Device to Host memcpy time: " << duration.count() << std::endl;

  t_start = high_resolution_clock::now();

	char* str1 = argv[1];
	char* str2 = "_nongds_output.txt";
	strcat(str1, str2);

	int o_fd = -1;
	o_fd = open(str1, O_CREAT | O_RDWR, 0664);
	if (o_fd < 0) {
		std::cerr << "file open error:" << std::endl;
	}

	printf("%d\n", o_fd);

  int cnt = 0;
	size_t alloc_size = query_nb * ref_nb * sizeof(float);
	int quo;
	size_t rem;
	ssize_t o_ret = -1;
  size_t test_offset = 2147479552 / 4;

  while(1){
    quo = alloc_size / 2147479552;
    printf("quo: %d\n", quo);
    if(quo > 0){
      o_ret = write(o_fd, dist+(cnt*test_offset), 2147479552);
      printf("%zd\n", o_ret);
      cnt++;
      alloc_size -= 2147479552;
      if (o_ret == -1) { 
        printf("Error when writing\n");
        break;
      }			
    }
    else{
      rem = alloc_size % 2147479552;
      o_ret = write(o_fd, dist+(cnt*test_offset), rem);
      printf("%zd\n", o_ret);
      break;
    }
  }

	close(o_fd);

	t_stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(t_stop - t_start);
	std::cout << "Tid: " << tid << " Host to Storage memcpy time: " << duration.count() << std::endl;

  t_start = high_resolution_clock::now();

  cudaFree(ind);
  cudaFree(dist);
  cudaFree(query);
  cudaFree(ref);

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

}
