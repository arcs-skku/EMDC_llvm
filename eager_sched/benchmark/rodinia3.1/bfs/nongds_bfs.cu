/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
 #include <stdlib.h>
 #include <stdio.h>
 #include <string.h>
 #include <math.h>
 #include <cuda.h>
 
 #include <bemps.hpp>
 #include <algorithm>
 #include <vector>
 #include <chrono>
 #include <iostream>
 
 #include "cufile.h"
 #include <fcntl.h>
 #include <assert.h>
 #include <unistd.h>
 
 using namespace std::chrono;
 
 #define MAX_THREADS_PER_BLOCK 512
 
 int no_of_nodes;
 int edge_list_size;
 FILE *fp;
 
 //Structure to hold a node information
 struct Node
 {
	 int starting;
	 int no_of_edges;
 };
 
 #include "kernel.cu"
 #include "kernel2.cu"
 
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
						if(loop < 5){
							CUDA_CHECK(cudaMemPrefetchAsync(ret.devPtr, ret.alloc_size, ret.device, s_e));
						}
						loop++;
					}
					// CUDA_CHECK(cudaStreamSynchronize(s_e));
					// CUDA_CHECK(cudaStreamDestroy(s_e));
					break;
				 }
			 }
		 }
	 }
 }
 
 void BFSGraph(int argc, char** argv);
 
 ////////////////////////////////////////////////////////////////////////////////
 // Main Program
 ////////////////////////////////////////////////////////////////////////////////
 int main( int argc, char** argv) 
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
 
	 no_of_nodes=0;
	 edge_list_size=201320478;
	 BFSGraph( argc, argv);
 
	 clock_gettime( CLOCK_REALTIME, &specific_time);
	 now = localtime(&specific_time.tv_sec);
	 millsec = specific_time.tv_nsec;
   
	 millsec = floor (specific_time.tv_nsec/1.0e6);
   
	 printf("Application end, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", 1900 + now->tm_year, 
		 now->tm_mon + 1, now->tm_mday, now->tm_hour, 
		 now->tm_min, now->tm_sec, millsec);
 }
 
 void Usage(int argc, char**argv){
 
 fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);
 
 }
 ////////////////////////////////////////////////////////////////////////////////
 //Apply BFS on a Graph using CUDA
 ////////////////////////////////////////////////////////////////////////////////
 void BFSGraph( int argc, char** argv) 
 {
	 int ef_cnt = 0;
	   int ret_dev_id;
	   int mem_intensive = 0;
	int wait_sign = 0;
	 int tid;
 
	 Parameters ret1;
	 Parameters ret2;
	 Parameters ret3;
	 Parameters ret4;
	 Parameters ret5;
	 Parameters ret6;
	 Parameters ret7;
 
	 // CUfileError_t status;
	 // CUfileDescr_t cf_descr;
	 // CUfileHandle_t cf_handle;
 
	 // status = cuFileDriverOpen();
	 // if (status.err != CU_FILE_SUCCESS) {
	 //         std::cerr << "cufile driver open error: "
	 // 	<< cuFileGetErrorString(status) << std::endl;
	 //         return -1;
	 // }
 
	 char *input_f;
	 if(argc!=3){
	 Usage(argc, argv);
	 exit(0);
	 }

	 input_f = argv[1];
	 printf("Reading File\n");
	 //Read in Graph from a file
	//  fp = fopen(input_f,"r");
	//  if(!fp)
	//  {
	// 	 printf("Error Reading graph file\n");
	// 	 return;
	//  }
 
	 tid = atoi(argv[2]);
 
	 int source = 0;
 
	//  fscanf(fp,"%d",&no_of_nodes);
 
	no_of_nodes = 33554432;
	 
	 int num_of_blocks = 1;
	 int num_of_threads_per_block = no_of_nodes;
 
	 // printf("%d\n", no_of_nodes);
 
	 //Make execution Parameters according to the number of nodes
	 //Distribute threads across multiple Blocks if necessary
	 if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	 {
		 num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		 num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	 }
 
	 // For reducing host initilization time in task
 
	Node* tmp_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	bool *tmp_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *tmp_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *tmp_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);
	int* tmp_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
 	int* tmp_cost = (int*) malloc( sizeof(int)*no_of_nodes);

	 //
 
	 dim3  grid( num_of_blocks, 1, 1);
	 dim3  threads( num_of_threads_per_block, 1, 1);
 
	 // printf("num_blocks: %d, num_threads: %d\n", num_of_blocks, num_of_threads_per_block);
 
	 size_t membytes = 0;
	 membytes += page_align(sizeof(Node)*no_of_nodes);
	 membytes += page_align(sizeof(int)*edge_list_size);
	 membytes += page_align(sizeof(bool)*no_of_nodes);
	 membytes += page_align(sizeof(bool)*no_of_nodes);
	 membytes += page_align(sizeof(bool)*no_of_nodes);
	 membytes += page_align(sizeof(int)*no_of_nodes);
	 membytes += page_align(sizeof(bool));
	 membytes += 309 * 1024 * 1024;
 
	 size_t max_k_usage = membytes;
	 size_t k_mem_req = membytes;

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
 
	//  if (membytes <= orig_alloc_mem){
	// 		full = 1;
	// 		priority = -5;
	// 	}
	// 	else{
	// 		full = 0;
	// 		priority = 0;
	// 	}

	if (max_k_usage <= orig_alloc_mem){
		full = 1;
		priority = -5;
	}
	else{
		full = 0;
		priority = 0;
	}

	printf("Full: %d, Wait sign: %d\n", full, wait_sign);
 
	   printf("ret_dev_id: %d\n", ret_dev_id);
 
	 auto t_start = high_resolution_clock::now();
 
	 // int id, cost, start, edgeno;
 
	 Node* h_graph_nodes;
	 CUDA_CHECK(cudaMallocManaged(&h_graph_nodes, sizeof(Node)*no_of_nodes));
 
	 bool *h_graph_mask;
	 CUDA_CHECK(cudaMallocManaged(&h_graph_mask, sizeof(bool)*no_of_nodes));
 
	 bool *h_updating_graph_mask;
	 CUDA_CHECK(cudaMallocManaged(&h_updating_graph_mask, sizeof(bool)*no_of_nodes));
 
	 bool *h_graph_visited;
	 CUDA_CHECK(cudaMallocManaged(&h_graph_visited, sizeof(bool)*no_of_nodes));
 
	 int* h_graph_edges;
	 CUDA_CHECK(cudaMallocManaged(&h_graph_edges, sizeof(int)*edge_list_size));
   
	 int* h_cost;
	 CUDA_CHECK(cudaMallocManaged(&h_cost, sizeof(int)*no_of_nodes));
 
	 bool *d_over;
	 CUDA_CHECK(cudaMallocManaged(&d_over, sizeof(bool)));
 
	 auto t_stop = high_resolution_clock::now();
	 auto duration = duration_cast<milliseconds>(t_stop - t_start);
	 std::cout << "Tid: " << tid << " Device mem allocation time: " << duration.count() << std::endl;
 
	 t_start = high_resolution_clock::now();
 
	 *d_over = true;
 
	 int fd = -1;
	 ssize_t ret = -1;
 
	 fd = open("node.txt", O_RDONLY, 0644);
	 printf("%d\n", fd);
 
	 ret = read(fd, h_graph_nodes, sizeof(Node)*no_of_nodes);
	 printf("%zd\n", ret);
	 close(fd);
 
	 fd = open("edges.txt", O_RDONLY, 0644);
	 printf("%d\n", fd);
 
	 ret = read(fd, h_graph_edges, sizeof(int)*edge_list_size);
	 printf("%zd\n", ret);
	 close(fd);
 
	 fd = open("mask.txt", O_RDONLY, 0644);
	 printf("%d\n", fd);
 
	 ret = read(fd, h_graph_mask, sizeof(bool)*no_of_nodes);
	 printf("%zd\n", ret);
	 close(fd);
 
	 fd = open("update_mask.txt", O_RDONLY, 0644);
	 printf("%d\n", fd);
 
	 ret = read(fd, h_updating_graph_mask, sizeof(bool)*no_of_nodes);
	 printf("%zd\n", ret);
	 close(fd);
 
	 fd = open("visited.txt", O_RDONLY, 0644);
	 printf("%d\n", fd);
 
	 ret = read(fd, h_graph_visited, sizeof(bool)*no_of_nodes);
	 printf("%zd\n", ret);
	 close(fd);
 
	 fd = open("cost.txt", O_RDONLY, 0644);
	 printf("%d\n", fd);
 
	 ret = read(fd, h_cost, sizeof(int)*no_of_nodes);
	 printf("%zd\n", ret);
	 close(fd);
 
	 
	 t_stop = high_resolution_clock::now();
	 duration = duration_cast<milliseconds>(t_stop - t_start);
	 std::cout << "Tid: " << tid << " Storage to Host memcpy time: " << duration.count() << std::endl;
 
	 ret1.devPtr = h_graph_nodes;
	 ret1.advice = cudaMemAdviseSetPreferredLocation;
	 ret1.device = ret_dev_id;
	 ret1.alloc_size = sizeof(Node)*no_of_nodes;
 
	   ret2.devPtr = h_graph_mask;
	 ret2.advice = cudaMemAdviseSetPreferredLocation;
	 ret2.device = ret_dev_id;
	 ret2.alloc_size = sizeof(bool)*no_of_nodes;
 
	   ret3.devPtr = h_updating_graph_mask;
	 ret3.advice = cudaMemAdviseSetPreferredLocation;
	 ret3.device = ret_dev_id;
	 ret3.alloc_size = sizeof(bool)*no_of_nodes;
 
	   ret4.devPtr = h_graph_visited;
	 ret4.advice = cudaMemAdviseSetPreferredLocation;
	 ret4.device = ret_dev_id;
	 ret4.alloc_size = sizeof(bool)*no_of_nodes;
 
	   ret5.devPtr = h_graph_edges;
	 ret5.advice = cudaMemAdviseSetPreferredLocation;
	 ret5.device = ret_dev_id;
	 ret5.alloc_size = sizeof(int)*edge_list_size;
 
	 ret6.devPtr = h_cost;
	 ret6.advice = cudaMemAdviseSetPreferredLocation;
	 ret6.device = ret_dev_id;
	 ret6.alloc_size = sizeof(int)*no_of_nodes;
 
	 ret7.devPtr = d_over;
	 ret7.advice = cudaMemAdviseSetPreferredLocation;
	 ret7.device = ret_dev_id;
	 ret7.alloc_size = sizeof(bool);
 
	 mem_list.push_back(ret1);
	 mem_list.push_back(ret2);
	 mem_list.push_back(ret3);
	 mem_list.push_back(ret4);
	 mem_list.push_back(ret5);
	 mem_list.push_back(ret6);
	 mem_list.push_back(ret7);
 
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
 
	 printf("Copied Everything to GPU memory\n");
 
	 int k=0;
	 printf("Start traversing the tree\n");
	 bool stop;
 
	//  if(full == 0){
	// 	el_wait(tid);
	// }
	int ready_to_launch = 0;
	// size_t os_110_dev = 1572864000;
	size_t os_110_dev = 786432000;
	// size_t os_110_dev = 3145728000;

	el_wait(tid, ready_to_launch);
	if((full == 0) && (!ready_to_launch)){
		el_wait(tid, ready_to_launch);
		u_mem = bemps_extra_task_mem(tid);
		// printf("%ld\n", u_mem);
		while((u_mem + os_110_dev) < k_mem_req){
			// chk_wait_sign(tid, wait_sign);
			u_mem = bemps_extra_task_mem(tid);
			el_wait(tid, ready_to_launch);

			if(((u_mem + os_110_dev) >= k_mem_req) || (ready_to_launch))
				break;
		}
		if(u_mem == max_k_usage){
			full = 1;
			priority = -5;
		}
		// checking 
	}
	// if((full == 0) && (!ready_to_launch)){
	// 	el_wait(tid, ready_to_launch);
	// 	u_mem = bemps_extra_task_mem(tid);
	// 	// printf("%ld\n", u_mem);
	// 	while(((float)u_mem * 1.1) < k_mem_req){
	// 		// chk_wait_sign(tid, wait_sign);
	// 		u_mem = bemps_extra_task_mem(tid);
	// 		el_wait(tid, ready_to_launch);

	// 		if((((float)u_mem * 1.1) >= k_mem_req) || (ready_to_launch))
	// 			break;
	// 	}
	// 	if(u_mem == max_k_usage){
	// 		full = 1;
	// 		priority = -5;
	// 	}
	// 	// checking 
	// }

	 t_start = high_resolution_clock::now();
 
	 //Call the Kernel untill all the elements of Frontier are not false
	 do
	 {
		 //if no thread changes this value then the loop stops
		 // stop=false;
		 // cudaMemcpy( d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) ;
		 *d_over = false;
		 
		 cudaStream_t s1;
		 CUDA_CHECK(cudaStreamCreateWithPriority(&s1, 0, priority));
		// CUDA_CHECK(cudaStreamCreate(&s1));
 
		 cudaEvent_t event1;
		 CUDA_CHECK(cudaEventCreateWithFlags(&event1, cudaEventDisableTiming));
 
		 Kernel<<< grid, threads, 0, s1 >>>( h_graph_nodes, h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost, no_of_nodes);
		 // check if kernel execution generated and error
		 
		 CUDA_CHECK(cudaEventRecord(event1, s1));
 
		 task_monitoring(event1, tid, orig_alloc_mem, membytes);
 
		 if(k == 0){
			launch_signal(tid);
		 }
		 
		 CUDA_CHECK(cudaStreamSynchronize(s1));
									 
		 CUDA_CHECK(cudaStreamDestroy(s1));
		 CUDA_CHECK(cudaEventDestroy(event1));
 
		 cudaStream_t s2;
		 CUDA_CHECK(cudaStreamCreateWithPriority(&s2, 0, priority));
		// CUDA_CHECK(cudaStreamCreate(&s2));
 
		 cudaEvent_t event2;
		 CUDA_CHECK(cudaEventCreateWithFlags(&event2, cudaEventDisableTiming));
 
		 Kernel2<<< grid, threads, 0, s2 >>>( h_graph_mask, h_updating_graph_mask, h_graph_visited, d_over, no_of_nodes);
		 // check if kernel execution generated and error
		 
		 CUDA_CHECK(cudaEventRecord(event2, s2));

		 task_monitoring(event2, tid, orig_alloc_mem, membytes);

		

		 CUDA_CHECK(cudaStreamSynchronize(s2));
									 
		 CUDA_CHECK(cudaStreamDestroy(s2));
		 CUDA_CHECK(cudaEventDestroy(event2));
 
		 // cudaMemcpy( &stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost) ;
		 k++;
	 }
	 while(*d_over);
 
	//  if(full == 1){
	// 	nl_signal(tid);
	// 	sig_cnt = 1;
	// }
 
	 t_stop = high_resolution_clock::now();
	 duration = duration_cast<milliseconds>(t_stop - t_start);
	 std::cout << "Tid: " << tid << " Kernel execution time: " << duration.count() << std::endl;
 
	 printf("Kernel Executed %d times\n",k);
 
	 // copy result from device to host
	 // cudaMemcpy( h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost) ;
 
	 t_start = high_resolution_clock::now();

	//  CUDA_CHECK(cudaMemPrefetchAsync(ret6.devPtr, ret6.alloc_size, cudaCpuDeviceId, 0));
	//  CUDA_CHECK(cudaDeviceSynchronize());
 
	CUDA_CHECK(cudaMemPrefetchAsync(ret6.devPtr, ret6.alloc_size, cudaCpuDeviceId, 0));

	 t_stop = high_resolution_clock::now();
	 duration = duration_cast<milliseconds>(t_stop - t_start);
	 std::cout << "Tid: " << tid << " Device to Host memcpy time: " << duration.count() << std::endl;

	//  memcpy(tmp_cost, h_cost, sizeof(int)*no_of_nodes);
		 
	 t_start = high_resolution_clock::now();

	 char* str1 = argv[2];
	 char* str2 = "_nongds_output.txt";
	 strcat(str1, str2);
 
	 int o_fd = -1;
	 o_fd = open(str1, O_CREAT | O_RDWR, 0664);
	 if (o_fd < 0) {
		 std::cerr << "file open error:" << std::endl;
	 }
 
	 printf("%d\n", o_fd);
 
	 ssize_t o_ret = -1;
	 o_ret = write(o_fd, tmp_cost, sizeof(int)*no_of_nodes);
	 
	 printf("%zd\n", o_ret);
 
	 close(o_fd);

	 //Store the result into a file
	//  FILE *fpo = fopen("result.txt","w");
	//  for(int i=0;i<no_of_nodes;i++)
	// 	 fprintf(fpo,"%d) cost:%d\n",i,tmp_cost[i]);
	//  fclose(fpo);
	//  printf("Result stored in result.txt\n");
 
	 t_stop = high_resolution_clock::now();
	 duration = duration_cast<milliseconds>(t_stop - t_start);
	 std::cout << "Tid: " << tid << " Host to Storage memcpy time: " << duration.count() << std::endl;
 
	//  if((sig_cnt == 0)){
	// 	u_mem = bemps_extra_task_mem(tid);
	// 	if (u_mem == membytes){    
	// 		nl_signal(tid);
	// 		sig_cnt = 1;
	// 		full = 1;
	// 	}
	// }

	 t_start = high_resolution_clock::now();
 
	 // cleanup memory
	 CUDA_CHECK(cudaFree( h_graph_nodes));
	 CUDA_CHECK(cudaFree( h_graph_edges));
	 CUDA_CHECK(cudaFree( h_graph_mask));
	 CUDA_CHECK(cudaFree( h_updating_graph_mask));
	 CUDA_CHECK(cudaFree( h_graph_visited));
	 CUDA_CHECK(cudaFree( h_cost));
	 CUDA_CHECK(cudaFree(d_over));
 
	 t_stop = high_resolution_clock::now();
	 duration = duration_cast<milliseconds>(t_stop - t_start);
	 std::cout << "Tid: " << tid << " Device memory deallocation time: " << duration.count() << std::endl;
 
	//  if((sig_cnt == 0)){
	// 	u_mem = bemps_extra_task_mem(tid);
	// 	if (u_mem == membytes){    
	// 		nl_signal(tid);
	// 		sig_cnt = 1;
	// 		full = 1;
	// 	}
	// }
	
	 bemps_free(tid);
 
	 clock_gettime( CLOCK_REALTIME, &specific_time);
	 now = localtime(&specific_time.tv_sec);
	 millsec = specific_time.tv_nsec;
 
	 millsec = floor (specific_time.tv_nsec/1.0e6);
 
	 printf("TID: %d finish work, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
		 now->tm_mon + 1, now->tm_mday, now->tm_hour, 
		 now->tm_min, now->tm_sec, millsec);
 }
 
