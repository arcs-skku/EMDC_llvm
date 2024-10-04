/* 
 * Copyright (c) 2009, Jiri Matela
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

 #include <unistd.h>
 #include <error.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <fcntl.h>
 #include <errno.h>
 #include <string.h>
 #include <assert.h>
 #include <sys/time.h>
 #include <getopt.h>
 
 #include "common.h"
 #include "components.h"
 #include "dwt.h"
 
 #include <bemps.hpp>
 #include <chrono>
 #include <iostream>
 #include <algorithm>
 #include <vector>
 #include <string>
 
 #include "cufile.h"
 
 using namespace std::chrono;
  
 #define CUDA_CHECK(val) { \
     if (val != cudaSuccess) { \
         fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(val), __LINE__, __FILE__); \
         exit(val); \
     } \
 }
 
 bool full; // 1 = fully secured
 int inputSize;
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
 
 struct dwt {
     char * srcFilename;
     char * outFilename;
     unsigned char *srcImg;
     int pixWidth;
     int pixHeight;
     int components;
     int dwtLvls;
 };
  
 int getImg(char * srcFilename, unsigned char *srcImg, int inputSize)
 {
     int i = -1;
 
     CUfileError_t status;
     CUfileDescr_t cf_descr;
     CUfileHandle_t cf_handle;
 
     status = cuFileDriverOpen();
     if (status.err != CU_FILE_SUCCESS) {
             std::cerr << "cufile driver open error: " << std::endl;
         // << cuFileGetErrorString(status) << std::endl;
             return -1;
     }
 
     // printf("Loading ipnput: %s\n", srcFilename);
     char *path = "../../data/dwt2d/";
     char *newSrc = NULL;
      
     if((newSrc = (char *)malloc(strlen(srcFilename)+strlen(path)+1)) != NULL)
     {
         newSrc[0] = '\0';
         strcat(newSrc, path);
         strcat(newSrc, srcFilename);
         srcFilename= newSrc;
     }
     printf("Loading input: %s\n", srcFilename);
  
     //srcFilename = strcat("../../data/dwt2d/",srcFilename);
     //read image
     i = open(srcFilename, O_RDONLY | O_DIRECT, 0644);
     if (i == -1) { 
         error(0,errno,"cannot access %s", srcFilename);
         return -1;
     }
 
     printf("Hello1\n");
 
     // fd = i;
 
     memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
     cf_descr.handle.fd = i;
     cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD; // linux based
     printf("%d\n", CU_FILE_HANDLE_TYPE_OPAQUE_FD);
     status = cuFileHandleRegister(&cf_handle, &cf_descr);
     printf("Hello3\n");
     if (status.err != CU_FILE_SUCCESS) {
         std::cerr << "file register error:" << std::endl;
             // << cuFileGetErrorString(status) << std::endl;
         close(i);
         i = -1;
         printf("Err1\n");
         return -1;
     }
 
     printf("Hello4\n");
 
     // int ret = read(i, srcImg, inputSize);
     // close(i);
  
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
 
    printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

     ssize_t ret = -1;
 
     ret = cuFileRead(cf_handle, srcImg, inputSize, 0, 0);
     printf("precteno %d, inputsize %d\n", ret, inputSize);
     if (ret < 0) {
         if (IS_CUFILE_ERR(ret))
             std::cerr << "read failed : " << std::endl;
                 // << cuFileGetErrorString(ret) << std::endl;
         else
             std::cerr << "read failed : " << std::endl;
                 // << cuFileGetErrorString(errno) << std::endl;
         cuFileHandleDeregister(cf_handle);
         close(i);
         printf("Err2\n");
         return -1;
     }
 
    cudaMemGetInfo(&free_mem, &total_mem);
 
    printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

     cuFileHandleDeregister(cf_handle);
     close (i);
 
     cuFileDriverClose();
 
     return 0;
 }
  
  
 void usage() {
     printf("dwt [otpions] src_img.rgb <out_img.dwt>\n\
    -d, --dimension\t\tdimensions of src img, e.g. 1920x1080\n\
    -c, --components\t\tnumber of color components, default 3\n\
    -b, --depth\t\t\tbit depth, default 8\n\
    -l, --level\t\t\tDWT level, default 3\n\
    -D, --device\t\t\tcuda device\n\
    -f, --forward\t\t\tforward transform\n\
    -r, --reverse\t\t\treverse transform\n\
    -9, --97\t\t\t9/7 transform\n\
    -5, --53\t\t\t5/3 transform\n\
    -w  --write-visual\t\twrite output in visual (tiled) fashion instead of the linear\n");
 }
  
 /* Store 3 RGB float components */
 __device__ void storeComponents(float *d_r, float *d_g, float *d_b, float r, float g, float b, int pos)
 {
     d_r[pos] = (r/255.0f) - 0.5f;
     d_g[pos] = (g/255.0f) - 0.5f;
     d_b[pos] = (b/255.0f) - 0.5f;
 }
  
  /* Store 3 RGB intege components */
 __device__ void storeComponents(int *d_r, int *d_g, int *d_b, int r, int g, int b, int pos)
 {
     d_r[pos] = r - 128;
     d_g[pos] = g - 128;
     d_b[pos] = b - 128;
 } 
 
 template <typename T>
 void init(T *arr, int pixWidth, int pixHeight){
     for(int i = 0; i < pixWidth * pixHeight * sizeof(T) / 4; i++){
         arr[i] = 0;
     }
 }
 
 /* Copy img src data into three separated component buffers */
 template<typename T>
 __global__ void c_CopySrcToComponents(T *d_r, T *d_g, T *d_b, 
                                    unsigned char * d_src, 
                                    int pixels)
 {
     int x  = threadIdx.x;
     int gX = blockDim.x*blockIdx.x;
  
     __shared__ unsigned char sData[256*3];
  
     /* Copy data to shared mem by 4bytes 
         other checks are not necessary, since 
         d_src buffer is aligned to sharedDataSize */
     if ( (x*4) < 256*3 ) {
         float *s = (float *)d_src;
         float *d = (float *)sData;
         d[x] = s[((gX*3)>>2) + x];
     }
     __syncthreads();
  
     T r, g, b;
  
     int offset = x*3;
     r = (T)(sData[offset]);
     g = (T)(sData[offset+1]);
     b = (T)(sData[offset+2]);
  
     int globalOutputPosition = gX + x;
     if (globalOutputPosition < pixels) {
         storeComponents(d_r, d_g, d_b, r, g, b, globalOutputPosition);
     }
 }
  
 template <typename T>
 void processDWT(struct dwt *d, int forward, int writeVisual, int tid)
 {
     int ret_dev_id;
     int ef_cnt = 0;
     size_t ef_mem;

     Parameters ret1;
     Parameters ret2;
     Parameters ret3;
     Parameters ret4;
     Parameters ret5;
     Parameters ret6;
     Parameters ret7;
     Parameters ret8;
 
     unsigned char * d_src;
  
     unsigned int componentSize = d->pixWidth*d->pixHeight*sizeof(T);
 
    //  T *tmp_c_r_out, *tmp_backup, *tmp_c_g_out, *tmp_c_b_out, *tmp_c_r, *tmp_c_g, *tmp_c_b;
 
    //  tmp_c_r_out = (T *)malloc(componentSize);
    //  tmp_backup = (T *)malloc(componentSize);
    //  tmp_c_g_out = (T *)malloc(componentSize);
    //  tmp_c_b_out = (T *)malloc(componentSize);
    //  tmp_c_r = (T *)malloc(componentSize);
    //  tmp_c_g = (T *)malloc(componentSize);
    //  tmp_c_b= (T *)malloc(componentSize);

    //  init(tmp_backup, d->pixWidth, d->pixHeight);
    //  init(tmp_c_r_out, d->pixWidth, d->pixHeight);
 
    //  init(tmp_c_g_out, d->pixWidth, d->pixHeight);
    //  init(tmp_c_b_out, d->pixWidth, d->pixHeight);
         
    //  init(tmp_c_r, d->pixWidth, d->pixHeight);
    //  init(tmp_c_g, d->pixWidth, d->pixHeight);
    //  init(tmp_c_b, d->pixWidth, d->pixHeight);
 
     int pixels      = d->pixWidth * d->pixHeight;
     int alignedSize =  DIVANDRND(pixels, 256) * 256 * 3;
 
     dim3 threads(256);
     dim3 grid(alignedSize/(256*3));
     assert(alignedSize%(256*3) == 0);
 
     // printf("grid_x: %d, grid_y: %d, thread_x: %d, thread_y: %d\n", grid.x, grid.y, threads.x, threads.y);
 
     size_t membytes = 0;
     membytes += page_align(componentSize);
     membytes += page_align(componentSize);
     membytes += page_align(componentSize);
     membytes += page_align(componentSize);
     membytes += page_align(componentSize);
     membytes += page_align(componentSize);
     membytes += page_align(componentSize);
     membytes += page_align(alignedSize);
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
 
     T *c_r_out, *backup ;
     T *c_g_out, *c_b_out;
    T *c_r, *c_g, *c_b;

    CUDA_CHECK(cudaMallocManaged(&c_r_out, componentSize)); //< aligned component size
    CUDA_CHECK(cudaMallocManaged(&backup, componentSize)); //< aligned component size
        
    /* Alloc two more buffers for G and B */
    CUDA_CHECK(cudaMallocManaged(&c_g_out, componentSize)); //< aligned component size
    CUDA_CHECK(cudaMallocManaged(&c_b_out, componentSize)); //< aligned component size

    /* Load components */
    CUDA_CHECK(cudaMallocManaged(&c_r, componentSize)); //< R, aligned component size
    CUDA_CHECK(cudaMallocManaged(&c_g, componentSize)); //< G, aligned component size
    CUDA_CHECK(cudaMallocManaged(&c_b, componentSize)); //< B, aligned component size

    CUDA_CHECK(cudaMallocManaged(&d_src, alignedSize));
     
     ret1.devPtr = c_r_out;
     ret1.advice = cudaMemAdviseSetPreferredLocation;
     ret1.device = ret_dev_id;
     ret1.alloc_size = componentSize;
 
     ret2.devPtr = backup;
     ret2.advice = cudaMemAdviseSetPreferredLocation;
     ret2.device = ret_dev_id;
     ret2.alloc_size = componentSize;
 
     if (d->components == 3) {
         ret3.devPtr = c_g_out;
         ret3.advice = cudaMemAdviseSetPreferredLocation;
         ret3.device = ret_dev_id;
         ret3.alloc_size = componentSize;
 
         ret4.devPtr = c_b_out;
         ret4.advice = cudaMemAdviseSetPreferredLocation;
         ret4.device = ret_dev_id;
         ret4.alloc_size = componentSize;
 
         ret5.devPtr = c_r;
         ret5.advice = cudaMemAdviseSetPreferredLocation;
         ret5.device = ret_dev_id;
         ret5.alloc_size = componentSize;
 
         ret6.devPtr = c_g;
         ret6.advice = cudaMemAdviseSetPreferredLocation;
         ret6.device = ret_dev_id;
         ret6.alloc_size = componentSize;
 
         ret7.devPtr = c_b;
         ret7.advice = cudaMemAdviseSetPreferredLocation;
         ret7.device = ret_dev_id;
         ret7.alloc_size = componentSize;
 
         ret8.devPtr = d_src;
         ret8.advice = cudaMemAdviseSetPreferredLocation;
         ret8.device = ret_dev_id;
         ret8.alloc_size = alignedSize;
 
         mem_list.push_back(ret1);
         mem_list.push_back(ret2);
         mem_list.push_back(ret3);
         mem_list.push_back(ret4);
         mem_list.push_back(ret5);
         mem_list.push_back(ret6);
         mem_list.push_back(ret7);
         mem_list.push_back(ret8);
 
         auto t_stop = high_resolution_clock::now();
         auto duration = duration_cast<milliseconds>(t_stop - t_start);
         std::cout << "Tid: " << tid << " Device mem allocation time: " << duration.count() << std::endl;
 
         if(full == 1){
            for(Parameters var : mem_list){
                CUDA_CHECK(cudaMemAdvise(var.devPtr, var.alloc_size, var.advice, var.device));
            }
        }
        
         t_start = high_resolution_clock::now();
 
        //  memset(backup, 0, componentSize);
        //  memset(c_r_out, 0, componentSize);
        //  memset(c_g_out, 0, componentSize);
        //  memset(c_b_out, 0, componentSize);
        //  memset(c_r, 0, componentSize);
        //  memset(c_g, 0, componentSize);
        //  memset(c_b, 0, componentSize);
 
         // inputSize = pixWidth*pixHeight*compCount; //<amount of data (in bytes) to proccess
         // d->srcImg = (unsigned char *)malloc(inputSize);
     
         size_t free_mem, total_mem;
        //  cudaMemGetInfo(&free_mem, &total_mem);
 
        //  printf("Free: %zd, Total: %zd\n", free_mem, total_mem);
 
         if (getImg(d->srcFilename, d_src, pixels*3) == -1) 
             printf("getImg Error!\n");
 
        // CUDA_CHECK(cudaDeviceSynchronize());
        
        //  cudaMemGetInfo(&free_mem, &total_mem);
 
        //  printf("Free: %zd, Total: %zd\n", free_mem, total_mem);
        //  printf("%d\n", pixels*3);
        //  printf("%d\n", alignedSize);
         // memcpy(d_src, d->srcImg, pixels*3);
 
         // init(backup, d->pixWidth, d->pixHeight);
         // init(c_r_out, d->pixWidth, d->pixHeight);
 
         // init(c_g_out, d->pixWidth, d->pixHeight);
         // init(c_b_out, d->pixWidth, d->pixHeight);
         
         // init(c_r, d->pixWidth, d->pixHeight);
         // init(c_g, d->pixWidth, d->pixHeight);
         // init(c_b, d->pixWidth, d->pixHeight);
 
         t_stop = high_resolution_clock::now();
         duration = duration_cast<milliseconds>(t_stop - t_start);
         std::cout << "Tid: " << tid << " Storage to Device memcpy time: " << duration.count() << std::endl;
 
         t_start = high_resolution_clock::now();
 
        CUDA_CHECK(cudaMemset(backup, 0, componentSize));
        CUDA_CHECK(cudaMemset(c_r_out, 0, componentSize));
        CUDA_CHECK(cudaMemset(c_g_out, 0, componentSize));
        CUDA_CHECK(cudaMemset(c_b_out, 0, componentSize));
        CUDA_CHECK(cudaMemset(c_r, 0, componentSize));
        CUDA_CHECK(cudaMemset(c_g, 0, componentSize));
        CUDA_CHECK(cudaMemset(c_b, 0, componentSize));

        //  if(full == 1){
        //     for(Parameters ret : mem_list){
        //         CUDA_CHECK(cudaMemAdvise(ret.devPtr, ret.alloc_size, ret.advice, ret.device));
        //     }
        //     CUDA_CHECK(cudaMemset(backup, 0, componentSize));
        //     CUDA_CHECK(cudaMemset(c_r_out, 0, componentSize));
        //     CUDA_CHECK(cudaMemset(c_g_out, 0, componentSize));
        //     CUDA_CHECK(cudaMemset(c_b_out, 0, componentSize));
        //     CUDA_CHECK(cudaMemset(c_r, 0, componentSize));
        //     CUDA_CHECK(cudaMemset(c_g, 0, componentSize));
        //     CUDA_CHECK(cudaMemset(c_b, 0, componentSize));
        //     // CUDA_CHECK(cudaMemPrefetchAsync(ret8.devPtr, ret8.alloc_size, ret8.device, 0));
        //  }
 
        //  CUDA_CHECK(cudaDeviceSynchronize());

        //  cudaMemGetInfo(&free_mem, &total_mem);
 
        //  printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

         t_stop = high_resolution_clock::now();
         duration = duration_cast<milliseconds>(t_stop - t_start);
         std::cout << "Tid: " << tid << " Host to Device memcpy time: " << duration.count() << std::endl;
 
         cudaEvent_t k_start, k_stop;
         CUDA_CHECK(cudaEventCreate(&k_start));
         CUDA_CHECK(cudaEventCreate(&k_stop));
         
         float total_k, time;
 
         cudaStream_t s1;
         CUDA_CHECK(cudaStreamCreateWithPriority(&s1, 0, priority));
         
         cudaEvent_t event1;
         CUDA_CHECK(cudaEventCreateWithFlags(&event1, cudaEventDisableTiming));
         
         CUDA_CHECK(cudaEventRecord(k_start));
 
         c_CopySrcToComponents<<<grid, threads, 0, s1>>>(c_r, c_g, c_b, d_src, pixels);
  
         CUDA_CHECK(cudaEventRecord(event1, s1));

         task_monitoring(event1, tid, orig_alloc_mem, membytes);
                                                   
         CUDA_CHECK(cudaEventRecord(k_stop));
         CUDA_CHECK(cudaEventSynchronize(k_stop));
         CUDA_CHECK(cudaEventElapsedTime(&total_k, k_start, k_stop));
                                                   
         printf("First_kernel: %f\n", total_k);
        
        CUDA_CHECK(cudaFree(d_src));

        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

        printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

        mem_list.erase(mem_list.begin()+7);
        
        if(full == 1){
            ef_mem = page_align(alignedSize);
            pre_bemps_free(tid, ef_mem);
            ef_cnt = 1;
        }

         /* Compute DWT and always store into file */
  
         t_start = high_resolution_clock::now();
 
         nStage2dDWT(c_r, c_r_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
         
         CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

  printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

  CUDA_CHECK(cudaFree(c_r));

  CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

  printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

  mem_list.erase(mem_list.begin()+4);

  if(full == 1){
    if(ef_cnt == 1){
      ef_mem = page_align(componentSize);
			pre_bemps_free(tid, ef_mem);
		}
		else{
      ef_mem = page_align(alignedSize) + page_align(componentSize);
			pre_bemps_free(tid, ef_mem);
			ef_cnt = 1;
		}
  }
         
         nStage2dDWT(c_g, c_g_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
         
         CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

         printf("Free: %zd, Total: %zd\n", free_mem, total_mem);
       
         CUDA_CHECK(cudaFree(c_g));
       
         CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
       
         printf("Free: %zd, Total: %zd\n", free_mem, total_mem);
       
         mem_list.erase(mem_list.begin()+4);

         if(full == 1){
           if(ef_cnt == 1){
             ef_mem = page_align(componentSize);
                   pre_bemps_free(tid, ef_mem);
               }
               else{
             ef_mem = page_align(alignedSize) + page_align(componentSize)*2;
                   pre_bemps_free(tid, ef_mem);
                   ef_cnt = 1;
               }
         }

         nStage2dDWT(c_b, c_b_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
       
        //  CUDA_CHECK(cudaDeviceSynchronize());

         CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

         printf("Free: %zd, Total: %zd\n", free_mem, total_mem);
       
         CUDA_CHECK(cudaFree(c_b));
         CUDA_CHECK(cudaFree(backup));
       
         CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
       
         printf("Free: %zd, Total: %zd\n", free_mem, total_mem);
       
         mem_list.erase(mem_list.begin()+4);
         mem_list.erase(mem_list.begin()+1);

         if(full == 1){
           if(ef_cnt == 1){
             ef_mem = page_align(componentSize)*2;
                   pre_bemps_free(tid, ef_mem);
               }
               else{
             ef_mem = page_align(alignedSize) + page_align(componentSize)*4;
                   pre_bemps_free(tid, ef_mem);
                   ef_cnt = 1;
               }
         }

         
         t_stop = high_resolution_clock::now();
         duration = duration_cast<milliseconds>(t_stop - t_start);
         std::cout << "\n\nTid: " << tid << " Kernel execution time: " << duration.count() << std::endl;
 
         // -------test----------
         // T *h_r_out=(T*)malloc(componentSize);
         // cudaMemcpy(h_r_out, c_g_out, componentSize, cudaMemcpyDeviceToHost);
         // int ii;
         // for(ii=0;ii<componentSize/sizeof(T);ii++) {
             // fprintf(stderr, "%d ", h_r_out[ii]);
             // if((ii+1) % (d->pixWidth) == 0) fprintf(stderr, "\n");
         // }
         // -------test----------
          
          
         /* Store DWT to file */
 #ifdef OUTPUT
         if (writeVisual) {
             printf("Output!!!!!\n!!!!!!\n!!!!!\n!!!!!\n");
             T *src_r, * src_g, *src_b;
             int samplesNum = d->pixWidth*d->pixHeight;
             int size = samplesNum*sizeof(T);
             CUDA_CHECK(cudaMallocHost((void **)&src_r, size));
             CUDA_CHECK(cudaMallocHost((void **)&src_g, size));
             CUDA_CHECK(cudaMallocHost((void **)&src_b, size));
             memset(src_r, 0, size);
             memset(src_g, 0, size);
             memset(src_b, 0, size);
             cudaMemcpy(src_r, c_r_out, size, cudaMemcpyDeviceToHost);
             cudaMemcpy(src_g, c_g_out, size, cudaMemcpyDeviceToHost);
             cudaMemcpy(src_b, c_b_out, size, cudaMemcpyDeviceToHost);

             cudaFree(c_r);
            cudaCheckError("Cuda free");
            cudaFree(c_g);
            cudaCheckError("Cuda free");
            cudaFree(c_b);
            cudaCheckError("Cuda free");
            cudaFree(c_g_out);
            cudaCheckError("Cuda free");
            cudaFree(c_b_out);
            cudaCheckError("Cuda free");
             cudaFree(c_r_out);
            cudaCheckError("Cuda free device");
            cudaFree(backup);
            cudaCheckError("Cuda free device");
            cudaFree(d_src);
            cudaCheckError("Cuda free device");
        
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
        
            printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

            bemps_free(tid);
        
            clock_gettime( CLOCK_REALTIME, &specific_time);
            now = localtime(&specific_time.tv_sec);
            millsec = specific_time.tv_nsec;
        
            millsec = floor (specific_time.tv_nsec/1.0e6);
        
            printf("TID: %d finish work, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
                now->tm_mon + 1, now->tm_mday, now->tm_hour, 
                now->tm_min, now->tm_sec, millsec);
         
             writeNStage2DDWT(src_r, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".r");
             writeNStage2DDWT(src_g, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".g");
             writeNStage2DDWT(src_b, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".b");
         } else {
            printf("Output without writeVisual!!\n");
             T *src_r, * src_g, *src_b;
             int samplesNum = d->pixWidth*d->pixHeight;
             int size = samplesNum*sizeof(T);
             CUDA_CHECK(cudaMallocHost((void **)&src_r, size));
             CUDA_CHECK(cudaMallocHost((void **)&src_g, size));
             CUDA_CHECK(cudaMallocHost((void **)&src_b, size));
             memset(src_r, 0, size);
             memset(src_g, 0, size);
             memset(src_b, 0, size);

             t_start = high_resolution_clock::now();

             CUDA_CHECK(cudaMemcpy(src_r, c_r_out, size, cudaMemcpyDeviceToHost));
             CUDA_CHECK(cudaMemcpy(src_g, c_g_out, size, cudaMemcpyDeviceToHost));
             CUDA_CHECK(cudaMemcpy(src_b, c_b_out, size, cudaMemcpyDeviceToHost));

             t_stop = high_resolution_clock::now();
            duration = duration_cast<milliseconds>(t_stop - t_start);
            std::cout << "Device to Host memcpy time: " << duration.count() << std::endl;

            t_start = high_resolution_clock::now();

            cudaFree(c_g_out);
            cudaCheckError("Cuda free");
            cudaFree(c_b_out);
            cudaCheckError("Cuda free");
             cudaFree(c_r_out);
            cudaCheckError("Cuda free device");
         
             t_stop = high_resolution_clock::now();
            duration = duration_cast<milliseconds>(t_stop - t_start);
            std::cout << "Memory deallocation time: " << duration.count() << std::endl;

             size_t free_mem, total_mem;
             cudaMemGetInfo(&free_mem, &total_mem);
         
             printf("Free: %zd, Total: %zd\n", free_mem, total_mem);
 
             bemps_free(tid);
         
             clock_gettime( CLOCK_REALTIME, &specific_time);
             now = localtime(&specific_time.tv_sec);
             millsec = specific_time.tv_nsec;
         
             millsec = floor (specific_time.tv_nsec/1.0e6);
         
             printf("TID: %d finish work, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
                 now->tm_mon + 1, now->tm_mday, now->tm_hour, 
                 now->tm_min, now->tm_sec, millsec);

             writeLinear(src_r, d->pixWidth, d->pixHeight, d->outFilename, ".r");
             writeLinear(src_g, d->pixWidth, d->pixHeight, d->outFilename, ".g");
             writeLinear(src_b, d->pixWidth, d->pixHeight, d->outFilename, ".b");
         }
 #endif
        //  cudaFree(c_r);
        //  cudaCheckError("Cuda free");
        //  cudaFree(c_g);
        //  cudaCheckError("Cuda free");
        //  cudaFree(c_b);
        //  cudaCheckError("Cuda free");
        //  cudaFree(c_g_out);
        //  cudaCheckError("Cuda free");
        //  cudaFree(c_b_out);
        //  cudaCheckError("Cuda free");
  
     } 
     else if (d->components == 1) {
 //         printf("Hello World\n");
 //          //Load component
 //          T *c_r;
 //          cudaMalloc((void**)&(c_r), componentSize); //< R, aligned component size
 //          cudaCheckError("Alloc device memory");
 //          cudaMemset(c_r, 0, componentSize);
 //          cudaCheckError("Memset device memory");
  
 //          bwToComponent(c_r, d->srcImg, d->pixWidth, d->pixHeight);
  
 //          // Compute DWT 
 //          nStage2dDWT(c_r, c_r_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
  
 //          // Store DWT to file 
 //  // #ifdef OUTPUT        
 //          if (writeVisual) {
 //              writeNStage2DDWT(c_r_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".out");
 //          } else {
 //              writeLinear(c_r_out, d->pixWidth, d->pixHeight, d->outFilename, ".lin.out");
 //          }
 //  // #endif
 //          cudaFree(c_r);
 //          cudaCheckError("Cuda free");
      }
  
    //  cudaFree(c_r_out);
    //  cudaCheckError("Cuda free device");
    //  cudaFree(backup);
    //  cudaCheckError("Cuda free device");
    //  cudaFree(d_src);
    //  cudaCheckError("Cuda free device");
 
    //  size_t free_mem, total_mem;
    //  cudaMemGetInfo(&free_mem, &total_mem);
 
    // printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

    //  bemps_free(tid);
 
    //  clock_gettime( CLOCK_REALTIME, &specific_time);
    //  now = localtime(&specific_time.tv_sec);
    //  millsec = specific_time.tv_nsec;
 
    //  millsec = floor (specific_time.tv_nsec/1.0e6);
 
    //  printf("TID: %d finish work, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
    //      now->tm_mon + 1, now->tm_mday, now->tm_hour, 
    //      now->tm_min, now->tm_sec, millsec);
 
 }
  
  int main(int argc, char **argv) 
  {
    int tid = atoi(argv[8]);
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
 
     int optindex = 0;
     char ch;
     struct option longopts[] = {
         {"dimension",   required_argument, 0, 'd'}, //dimensions of src img
         {"components",  required_argument, 0, 'c'}, //numger of components of src img
         {"depth",       required_argument, 0, 'b'}, //bit depth of src img
         {"level",       required_argument, 0, 'l'}, //level of dwt
         {"device",      required_argument, 0, 'D'}, //cuda device
         {"forward",     no_argument,       0, 'f'}, //forward transform
         {"reverse",     no_argument,       0, 'r'}, //reverse transform
         {"97",          no_argument,       0, '9'}, //9/7 transform
         {"53",          no_argument,       0, '5' }, //5/3transform
         {"write-visual",no_argument,       0, 'w' }, //write output (subbands) in visual (tiled) order instead of linear
         {"help",        no_argument,       0, 'h'}  
     };
      
     int pixWidth    = 0; //<real pixWidth
     int pixHeight   = 0; //<real pixHeight
     int compCount   = 3; //number of components; 3 for RGB or YUV, 4 for RGBA
     int bitDepth    = 8; 
     int dwtLvls     = 3; //default numuber of DWT levels
     int device      = 0;
     int forward     = 1; //forward transform
     int dwt97       = 1; //1=dwt9/7, 0=dwt5/3 transform
     int writeVisual = 0; //write output (subbands) in visual (tiled) order instead of linear
     char * pos;
  
     while ((ch = getopt_long(argc, argv, "d:c:b:l:D:fr95wh", longopts, &optindex)) != -1) {
         switch (ch) {
         case 'd':
             pixWidth = atoi(optarg);
             pos = strstr(optarg, "x");
             if (pos == NULL || pixWidth == 0 || (strlen(pos) >= strlen(optarg))) {
                 usage();
                 return -1;
             }
             pixHeight = atoi(pos+1);
             break;
         case 'c':
             compCount = atoi(optarg);
             break;
         case 'b':
             bitDepth = atoi(optarg);
             break;
         case 'l':
             dwtLvls = atoi(optarg);
             break;
         case 'D':
             device = atoi(optarg);
             break;
         case 'f':
             forward = 1;
             break;
         case 'r':
             forward = 0;
             break;
         case '9':
             dwt97 = 1;
             break;
         case '5':
             dwt97 = 0;
             break;
         case 'w':
             writeVisual = 1;
             break;
         case 'h':
             usage();
             return 0;
         case '?':
             return -1;
         default :
             usage();
             return -1;
         }
     }
  
     argc -= optind;
     argv += optind;
  
     if (argc == 0) { // at least one filename is expected
         printf("Please supply src file name\n");
         usage();
         return -1;
     }
  
     if (pixWidth <= 0 || pixHeight <=0) {
         printf("Wrong or missing dimensions\n");
         usage();
         return -1;
     }
  
     if (forward == 0) {
         writeVisual = 0; //do not write visual when RDWT
     }
  
     struct dwt *d;
     d = (struct dwt *)malloc(sizeof(struct dwt));
     d->srcImg = NULL;
     d->pixWidth = pixWidth;
     d->pixHeight = pixHeight;
     d->components = compCount;
     d->dwtLvls  = dwtLvls;
 
     // file names
     d->srcFilename = (char *)malloc(strlen(argv[0]));
     strcpy(d->srcFilename, argv[0]);
     if (argc == 1) { // only one filename supplyed
         d->outFilename = (char *)malloc(strlen(d->srcFilename)+4);
         strcpy(d->outFilename, d->srcFilename);
         strcpy(d->outFilename+strlen(d->srcFilename), ".dwt");
     } else {
         d->outFilename = strdup(argv[1]);
     }
 
     //Input review
     printf("Source file:\t\t%s\n", d->srcFilename);
     printf(" Dimensions:\t\t%dx%d\n", pixWidth, pixHeight);
     printf(" Components count:\t%d\n", compCount);
     printf(" Bit depth:\t\t%d\n", bitDepth);
     printf(" DWT levels:\t\t%d\n", dwtLvls);
     printf(" Forward transform:\t%d\n", forward);
     printf(" 9/7 transform:\t\t%d\n", dwt97);
      
     //data sizes
     // inputSize = pixWidth*pixHeight*compCount; //<amount of data (in bytes) to proccess
     
     // unsigned int componentSize = d->pixWidth*d->pixHeight*sizeof(int);
 
     // int pixels      = d->pixWidth * d->pixHeight;
     // int alignedSize =  DIVANDRND(pixels, 256) * 256 * 3;
 
     // dim3 threads(256);
     // dim3 grid(alignedSize/(256*3));
     // assert(alignedSize%(256*3) == 0);
 
     // size_t membytes = 0;
     // membytes += page_align(componentSize);
     // membytes += page_align(componentSize);
     // membytes += page_align(componentSize);
     // membytes += page_align(componentSize);
     // membytes += page_align(componentSize);
     // membytes += page_align(componentSize);
     // membytes += page_align(componentSize);
     // membytes += page_align(alignedSize);
       // membytes += 309 * 1024 * 1024;
 
     // clock_gettime( CLOCK_REALTIME, &specific_time);
     // now = localtime(&specific_time.tv_sec);
     // millsec = specific_time.tv_nsec;
   
     // millsec = floor (specific_time.tv_nsec/1.0e6);
   
     // printf("TID: %d before schedule, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
     //     now->tm_mon + 1, now->tm_mday, now->tm_hour, 
     //     now->tm_min, now->tm_sec, millsec);
       
     // long orig_alloc_mem = bemps_begin(tid, grid.x, grid.y, grid.z, threads.x, threads.y, threads.z, membytes, ret_dev_id);
       
     // clock_gettime( CLOCK_REALTIME, &specific_time);
     // now = localtime(&specific_time.tv_sec);
     // millsec = specific_time.tv_nsec;
   
     // millsec = floor (specific_time.tv_nsec/1.0e6);
   
     // printf("TID: %d after schedule, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
     //     now->tm_mon + 1, now->tm_mday, now->tm_hour, 
     //     now->tm_min, now->tm_sec, millsec);
       
     // if (membytes <= orig_alloc_mem)
     //     full = 1;
     // else
     //     full = 0;
     // printf("Full: %d\n", full);
   
     // printf("ret_dev_id: %d\n", ret_dev_id);
 
     //load img source image
     // CUDA_CHECK(cudaMallocManaged(&d->srcImg, inputSize))
 
     // inputSize = pixWidth*pixHeight*compCount; //<amount of data (in bytes) to proccess
     // d->srcImg = (unsigned char *)malloc(inputSize);
     
     // if (getImg(d->srcFilename, d->srcImg, inputSize) == -1) 
     //     return -1;
  
      /* DWT */
     if (forward == 1) {
         if(dwt97 == 1 )
             processDWT<float>(d, forward, writeVisual, tid);
         else // 5/3
             processDWT<int>(d, forward, writeVisual, tid);
     }
     else { // reverse
         if(dwt97 == 1 )
             processDWT<float>(d, forward, writeVisual, tid);
         else // 5/3
             processDWT<int>(d, forward, writeVisual, tid);
     }
  
     //writeComponent(r_cuda, pixWidth, pixHeight, srcFilename, ".g");
     //writeComponent(g_wave_cuda, 512000, ".g");
     //writeComponent(g_cuda, componentSize, ".g");
     //writeComponent(b_wave_cuda, componentSize, ".b");
     // cudaFreeHost(d->srcImg);
     // cudaCheckError("Cuda free host");
     
     free(d->srcImg);
 
     clock_gettime( CLOCK_REALTIME, &specific_time);
    now = localtime(&specific_time.tv_sec);
    millsec = specific_time.tv_nsec;
      
    millsec = floor (specific_time.tv_nsec/1.0e6);
      
    printf("TID: %d Application end, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", tid, 1900 + now->tm_year, 
        now->tm_mon + 1, now->tm_mday, now->tm_hour, 
        now->tm_min, now->tm_sec, millsec);
             
     return 0;
 }
  
