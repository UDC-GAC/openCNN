// Copyright 2021 Roberto Lopez Castro
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "FX_m2.cu"

#ifdef OPTLDS64
#include "store_and_transform_output_optLDS64.cuh"
#include "outer_product.cuh"
#elif OPTSTS64_CMP
#include "store_and_transform_output_optSTS64_compact.cuh"
#include "outer_product_suffle.cuh"
#else
#include "store_and_transform_output_optSTS64.cuh"
#include "outer_product_suffle.cuh"
#endif

#ifdef _noWALL_
typedef struct rusage resnfo;
typedef struct _timenfo {
  double time;
  double systime;
} timenfo;
#define timestamp(sample) getrusage(RUSAGE_SELF, (sample))
#define printtime(t) printf("%15f s (%f user + %f sys) ",		\
			    t.time + t.systime, t.time, t.systime);
#else
typedef struct timeval resnfo;
typedef double timenfo;
#define timestamp(sample)     gettimeofday((sample), 0)
#define printtime(t) printf("%15f s ", t);
#endif

#ifndef _WINOGRAD_
#define _WINOGRAD_
extern "C"
{


#define d(input, i, j) ( input[(i<<2) + (j)] )

__device__ __forceinline__ void load_and_transform_input_tile(float *Btd, float *pOutputs, int in_h, int in_w,
                                  int tiles_dim, int in_c, int in_n, int tile_size, 
                                  int tiles_2d_dim, int tile_2d_s, int Inx, int Iny, int TileX, int TileY){

  float workspace[3];
  
  #pragma unroll
  for(int j=0; j<4; j++){
    workspace[0] = Btd[j];
    workspace[1] = Btd[j+4];
    workspace[2] = Btd[j+8];

    Btd[j]    = workspace[0] - workspace[2];
    Btd[j+4]  = workspace[1] + workspace[2];
    Btd[j+8]  = workspace[2] - workspace[1];
    Btd[j+12] = workspace[1] - Btd[j+12];
  }
  
  int c_offset = BN*BC;
  int c_tensor = Iny*BN + Inx;
  
  #pragma unroll
  for(int i=0; i<4; i++){ // prefetch 1 input tile/thread
    pOutputs[c_tensor+i*c_offset*4] = d(Btd, i, 0) - d(Btd, i, 2);  
    pOutputs[c_tensor+i*c_offset*4+c_offset] = d(Btd, i, 1) + d(Btd, i, 2);
    pOutputs[c_tensor+i*c_offset*4+2*c_offset] = d(Btd, i, 2) - d(Btd, i, 1);
    pOutputs[c_tensor+i*c_offset*4+3*c_offset] = d(Btd, i, 1) - d(Btd, i, 3);
  }     

}

__device__ __forceinline__ void load_filter_tile(float *tiles, float *pOutputs, 
                                int filt_c, int filt_k, int Inx, int Iny){
 
  int c_tensor_s = Iny*BK + Inx;
  int c_offset_s = BK*BC;
  
  for(int k=0; k<2; k++){ // prefetch 2 filter tiles/thread
    for(int i=0; i<4; i++){
      for(int j=0; j<4; j++){
        pOutputs[c_tensor_s + i*c_offset_s*4 + j*c_offset_s] = tiles[k*16 + i*4 + j];
      }
    }

    c_tensor_s += BN;
  }
  
}

__device__ __forceinline__ void prefetch_filter_tile(float *pInputs, float *tiles, 
                                      int filt_k, int Inx, int Iny, int TileZ){

  int c_tensor = TileZ*BK + (Iny*filt_k<<4) + Inx;
  
  int acumm;
  #pragma unroll  
  for(int i=0; i<4; i++){
      acumm = (i*filt_k<<2);
      for(int j=0; j<4; j++){
          tiles[(i<<2) + j] = pInputs[acumm + j*filt_k + c_tensor];
          tiles[16 + (i<<2) + j] = pInputs[acumm + j*filt_k + c_tensor+BN];
      }
  }
}

__device__ __forceinline__ void prefetch_input_tile(float *pInputs, float *tile, int in_h, int in_w,
                  int in_n, int Inx, int Iny, int TileX, int TileY, int tiles_dim, short mask){
  
  int c_tensor = (TileY%tiles_dim)*in_n*2 + (TileY/tiles_dim)*in_n*in_w*2 + TileX*BN + Iny*(in_n*in_h*in_w) + (Inx/in_n)*2*in_n + (Inx%in_n) - (in_n*in_w+in_n);
  int acumm,x;
           
  if(mask==0xFFFF){

    for(int i=0; i<4; i++){
      acumm = i*in_n*in_w;   
      #pragma unroll
      for(int j=0; j<4; j++){
        tile[(i<<2) + j] = pInputs[acumm + j*in_n + c_tensor];
      }
    }

  } else {

     for(int i=0; i<4; i++){
      acumm = i*in_n*in_w;   
      #pragma unroll
      for(int j=0; j<4; j++){
        x = (i<<2) + j;
        tile[x] = 0;
        if(mask&(1<<x))
          tile[x]=pInputs[acumm + j*in_n + c_tensor];
      }
    }
  }
}

__device__  __forceinline__ void prefetch_filter_frag(float4 *filter_frag, float4 *B_frag, int f_frag_offset, 
                          int Inx, int offset1, int offset2){

  *((float4*) (filter_frag))     = *(B_frag + offset1);
  *((float4*) (filter_frag + 1)) = *(B_frag + offset2);

  *((float4*) (filter_frag + 2)) = *(B_frag + f_frag_offset + offset1);
  *((float4*) (filter_frag + 3)) = *(B_frag + f_frag_offset + offset2);
}

__device__  __forceinline__ void prefetch_input_frag(float4* input_frag, float4 *A_frag, int frag_offset, 
                      int Inx, int offset1, int offset2){  
  
  *((float4*) (input_frag))     = *(A_frag + offset1); //ld_shared(A_frag + offset1);
  *((float4*) (input_frag + 1)) = *(A_frag + offset2);

  *((float4*) (input_frag + 2)) = *(A_frag + frag_offset + offset1);
  *((float4*) (input_frag + 3)) = *(A_frag + frag_offset + offset2); //3=2+1
}

__global__ void Winograd_kernel(float *A, float *B, float *C,
                    int tiles_dim, int in_c, int in_n, int in_h, int in_w, 
                    int tile_size, int filt_k, int filt_c,
                    int tiles_2d_dim, int out_c, int out_n, 
                    int tile_2d_s, int out_h, int out_w){

  extern __shared__ float shared_mem[];
  float *input_smem  = (float*)shared_mem;
  float *filter_smem = (float*)&shared_mem[16*BC*BN];

  short m = 0xFFFF;
  if((blockIdx.y/tiles_dim)==0)   m&=0xFFF0;
  if((blockIdx.y/tiles_dim)==(tiles_dim-1)) m &= (!(in_w%2))?(0x0FFF):(0x00FF);
  if(!((blockIdx.y+1)%tiles_dim)) m &= (!(in_w%2))?(0x7777):(0x3333);
  if(!((blockIdx.y)%tiles_dim))   m&=0xeeee;

  float img_tile[16]; // Prefetch input from GMEM
  float filter_tile[32]; // Prefetch filter from GMEM

  float4 input_frag_mem[8];  //2*2(2*8/4) Data to do Outer Product + prefetch f. SMEM (double_buffer)
  float4 filter_frag_mem[8]; //2*2 Data to do Outer Product + prefetch f. SMEM (double_buffer)
  float4 accumulator[2][16] = {0.0f};  // Accumulators 

  float4 *A_frag; // Input data pointer
  int frag_offset = 2* (BC*BN); // (2=8/4) SMEM input read offset

  float4 *B_frag; // Filter data pointer
  int f_frag_offset = 2* (BC*BK); // (2=8/4) SMEM filter read offset

  float4 *input_frag  = (float4*) input_frag_mem;
  float4 *filter_frag = (float4*) filter_frag_mem;

  float4 *swap;

  prefetch_input_tile(A, img_tile, in_h, in_w, in_n, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, tiles_dim, m);
  prefetch_filter_tile(B, filter_tile, filt_k, threadIdx.x, threadIdx.y, blockIdx.z);

  float4 *input_frag_buffer  = (float4*) (input_frag+4);
  float4 *filter_frag_buffer = (float4*) (filter_frag+4);
  
  // Mainloop - iterates over the entire K dimension - not unrolled
  for(int iter=0; iter<in_c; iter+=BC){ // Current iteration

    A_frag = (float4*) (input_smem  + threadIdx.y*BC*BN);
    B_frag = (float4*) (filter_smem + threadIdx.y*BC*BK);

    load_and_transform_input_tile(img_tile, input_smem, in_h, in_w,
                 tiles_dim, in_c, in_n, tile_size,
                 tiles_2d_dim, tile_2d_s, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
    load_filter_tile(filter_tile, filter_smem, filt_c, filt_k, threadIdx.x, threadIdx.y);

    __syncthreads();

    prefetch_input_frag(input_frag, A_frag, frag_offset, threadIdx.x, access_s[0][threadIdx.x], access_s[1][threadIdx.x]);
    prefetch_filter_frag(filter_frag, B_frag, f_frag_offset, threadIdx.x, access_f_s[0][threadIdx.x], access_f_s[1][threadIdx.x]);
    
    #pragma unroll
    for(int i=0; i<BC; i++){

      if(i<(BC-1)){
        A_frag += BN/4;
        B_frag += BK/4;

        prefetch_input_frag(input_frag_buffer, A_frag, frag_offset, threadIdx.x, access_s[0][threadIdx.x], access_s[1][threadIdx.x]);
        prefetch_filter_frag(filter_frag_buffer, B_frag, f_frag_offset, threadIdx.x, access_f_s[0][threadIdx.x], access_f_s[1][threadIdx.x]);
      }

      outer_product(input_frag, filter_frag, accumulator);

      swap = input_frag;
      input_frag = input_frag_buffer;
      input_frag_buffer = swap;

      swap = filter_frag;
      filter_frag = filter_frag_buffer;
      filter_frag_buffer = swap;
      
    }
    
    A += in_n*BC*in_w*in_h;
    B += filt_k*BC*4*4;

    if(iter<(in_c-BC)){
      prefetch_input_tile(A, img_tile, in_h, in_w, in_n, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, tiles_dim, m);
      prefetch_filter_tile(B, filter_tile, filt_k, threadIdx.x, threadIdx.y, blockIdx.z);
    }

    __syncthreads();
  }

  // Transpose, transform and store accumulated result
  store_output_tile(accumulator, shared_mem, threadIdx.x, threadIdx.y, C, blockIdx.x,blockIdx.y, blockIdx.z, out_h, out_w, tiles_dim, out_n, input_frag_mem, filter_frag_mem, m);
                     
}

cudaError_t convolutionForward_32x64x8(float *k, int in_h, int in_w, float *w, int out_h,
                  int out_w, int out_n, int out_c, float *C, float *Ww, 
                const unsigned int n,
                int tiles_dim, int in_n, int tile_size,
                int in_c, int filt_k, int filt_c, int filt_h, int filt_w, int alpha, int m){

  int tile_2d_s = tile_size*tile_size;
  int tiles_2d_dim = tiles_dim*tiles_dim;
  int smem_size = (16*BC*BN + 16*BC*BK)*4;

  FX<<<dim3(filt_k/BK, filt_c/BC), dim3(BN, BC)>>>(w, Ww, filt_k, filt_c, filt_h, filt_w, alpha);
        
  #ifdef OPTSTS64_CMP
  smem_size = 65536; // 64 KB
  cudaFuncSetAttribute(Winograd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  #endif

  Winograd_kernel<<<dim3(in_n/BN, tiles_2d_dim, filt_k/BK), dim3(BN, 8), smem_size>>>(k, Ww, C, tiles_dim, in_c, in_n, in_h, in_w, tile_size, filt_k, filt_c, tiles_2d_dim, out_c, out_n, tile_2d_s, out_h, out_w);

  return cudaGetLastError();
}

}
#endif
