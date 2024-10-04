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

#include <chrono>
#include <iostream>

using namespace std::chrono;

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
    printf("Loading ipnput: %s\n", srcFilename);
 
    //srcFilename = strcat("../../data/dwt2d/",srcFilename);
    //read image
    int i = open(srcFilename, O_RDONLY, 0644);
    if (i == -1) { 
        error(0,errno,"cannot access %s", srcFilename);
        return -1;
    }
    int ret = read(i, srcImg, inputSize);
    printf("precteno %d, inputsize %d\n", ret, inputSize);
    close(i);
 
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
    unsigned char * d_src;

    int os_perc = tid;
 
    int componentSize = d->pixWidth*d->pixHeight*sizeof(T);
 
    int pixels      = d->pixWidth * d->pixHeight;
    int alignedSize =  DIVANDRND(pixels, 256) * 256 * 3;
 
    printf("Component Size: %d\n", componentSize);
    
    // size_t mem_req = 2135949312;
    // size_t mem_req = 7571767296;

    size_t k1_mem_req = 4026531840;
    size_t k2_mem_req = 6174015488;
    size_t k3_mem_req = 7247757312;
    size_t k4_mem_req = 8321499136;

    size_t f_free_mem, f_total_mem;
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&f_free_mem, &f_total_mem));
 
    printf("Free: %zd, Total: %zd\n", free_mem, total_mem);
 
    float dummy_perc = ((float)os_perc / (float)100);
	printf("Dummy percentage: %f\n", dummy_perc);

	int* dummy;
	size_t dummy_size;
	// printf("Dummy size: %zd\n", dummy_size);

	// CUDA_CHECK(cudaMalloc((void**) &dummy, dummy_size))
	// CUDA_CHECK(cudaMemset(dummy, 0, dummy_size));

    dim3 threads(256);
    dim3 grid(alignedSize/(256*3));
    assert(alignedSize%(256*3) == 0);
 
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
 
    T *c_r_out, *backup ;
    CUDA_CHECK(cudaMallocManaged(&c_r_out, componentSize)); //< aligned component size
    // cudaMemset(c_r_out, 0, componentSize);
     
    CUDA_CHECK(cudaMallocManaged(&backup, componentSize)); //< aligned component size
    // CUDA_CHECK(cudaMemset(backup, 0, componentSize));

    if (d->components == 3) {
         /* Alloc two more buffers for G and B */
        T *c_g_out, *c_b_out;
        CUDA_CHECK(cudaMallocManaged(&c_g_out, componentSize)); //< aligned component size
        // CUDA_CHECK(cudaMemset(c_g_out, 0, componentSize));
         
        CUDA_CHECK(cudaMallocManaged(&c_b_out, componentSize)); //< aligned component size
        // CUDA_CHECK(cudaMemset(c_b_out, 0, componentSize));
         
         /* Load components */
        T *c_r, *c_g, *c_b;
        CUDA_CHECK(cudaMallocManaged(&c_r, componentSize)); //< R, aligned component size
        // CUDA_CHECK(cudaMemset(c_r, 0, componentSize));

        CUDA_CHECK(cudaMallocManaged(&c_g, componentSize)); //< G, aligned component size
        // CUDA_CHECK(cudaMemset(c_g, 0, componentSize));

        CUDA_CHECK(cudaMallocManaged(&c_b, componentSize)); //< B, aligned component size
        // CUDA_CHECK(cudaMemset(c_b, 0, componentSize));
 
        float total_k, time;

        // CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
 
        // printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

        dummy_size = page_align(f_free_mem * dummy_perc) - page_align(k1_mem_req);
        printf("Dummy size: %zd\n", dummy_size);

        CUDA_CHECK(cudaMalloc((void**) &dummy, dummy_size))
        CUDA_CHECK(cudaMemset(dummy, 0, dummy_size));

        cudaEvent_t k_start, k_stop;
        CUDA_CHECK(cudaEventCreate(&k_start));
        CUDA_CHECK(cudaEventCreate(&k_stop));

        CUDA_CHECK(cudaEventRecord(k_start));

        c_CopySrcToComponents<<<grid, threads>>>(c_r, c_g, c_b, d->srcImg, pixels);
 
        CUDA_CHECK(cudaEventRecord(k_stop));
        CUDA_CHECK(cudaEventSynchronize(k_stop));
        CUDA_CHECK(cudaEventElapsedTime(&total_k, k_start, k_stop));
                                                   
        printf("First_kernel: %f\n", total_k);

        // CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
 
        // printf("Free: %zd, Total: %zd\n", free_mem, total_mem);

        /* Compute DWT and always store into file */
 
        cudaFree(dummy);

        dummy_size = f_free_mem - page_align(k2_mem_req * dummy_perc);

        CUDA_CHECK(cudaMalloc((void**) &dummy, dummy_size))
        CUDA_CHECK(cudaMemset(dummy, 0, dummy_size));

        auto t_start = high_resolution_clock::now();
        
        nStage2dDWT(c_r, c_r_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
        
        auto t_stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(t_stop - t_start);
        std::cout << "Kernel execution time: " << duration.count() << std::endl;

        cudaFree(dummy);

        dummy_size = f_free_mem - page_align(k3_mem_req * dummy_perc);

        CUDA_CHECK(cudaMalloc((void**) &dummy, dummy_size))
        CUDA_CHECK(cudaMemset(dummy, 0, dummy_size));

        t_start = high_resolution_clock::now();

        nStage2dDWT(c_g, c_g_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
        
        t_stop = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(t_stop - t_start);
        std::cout << "Kernel execution time: " << duration.count() << std::endl;

        cudaFree(dummy);

        dummy_size = f_free_mem - page_align(k4_mem_req * dummy_perc);

        CUDA_CHECK(cudaMalloc((void**) &dummy, dummy_size))
        CUDA_CHECK(cudaMemset(dummy, 0, dummy_size));

        t_start = high_resolution_clock::now();

        nStage2dDWT(c_b, c_b_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
      
        t_stop = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(t_stop - t_start);
        std::cout << "Kernel execution time: " << duration.count() << std::endl;

        cudaFree(dummy);

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
            writeNStage2DDWT(c_r_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".r");
            writeNStage2DDWT(c_g_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".g");
            writeNStage2DDWT(c_b_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".b");
        } else {
            writeLinear(c_r_out, d->pixWidth, d->pixHeight, d->outFilename, ".r");
            writeLinear(c_g_out, d->pixWidth, d->pixHeight, d->outFilename, ".g");
            writeLinear(c_b_out, d->pixWidth, d->pixHeight, d->outFilename, ".b");
        }
#endif
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
 
    cudaFree(c_r_out);
    cudaCheckError("Cuda free device");
    cudaFree(backup);
    cudaCheckError("Cuda free device");
 
    clock_gettime( CLOCK_REALTIME, &specific_time);
    now = localtime(&specific_time.tv_sec);
    millsec = specific_time.tv_nsec;
 
    millsec = floor (specific_time.tv_nsec/1.0e6);
 
 
    printf("Task finish, [%04d/%02d/%02d] %02d:%02d:%02d msec : %d\n", 1900 + now->tm_year, 
        now->tm_mon + 1, now->tm_mday, now->tm_hour, 
        now->tm_min, now->tm_sec, millsec);
 
}
 
 int main(int argc, char **argv) 
 {
    int tid;
    tid = atoi(argv[8]);
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
    int inputSize = pixWidth*pixHeight*compCount; //<amount of data (in bytes) to proccess
     
    //load img source image
    CUDA_CHECK(cudaMallocManaged(&d->srcImg, inputSize))
    
    if (getImg(d->srcFilename, d->srcImg, inputSize) == -1) 
        return -1;
 
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
    
    CUDA_CHECK(cudaFree(d->srcImg));
 
    return 0;
}
