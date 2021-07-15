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

#include "../config.hpp"

#ifndef _OUTPUT_KERNEL_PPoPP_
#define _OUTPUT_KERNEL_PPoPP_
extern "C"
{

__device__ void  transform_output_tile(float *pOutputs, float *C_tile, float *At, int out_h, int out_w, 
                                      int tiles_dim, int round, int in_n, int offset, int out_thread[][4], short mask, int c_tensor, int c_glb_offset){

  for(int j=0; j<4; j++){
    At[j] = C_tile[j] + C_tile[4+j] + C_tile[8+j];
    At[j+8] = C_tile[j+16] + C_tile[4+j+16] + C_tile[8+j+16];
    
    At[4+j] = C_tile[4+j] - C_tile[8+j] - C_tile[12+j];
    At[4+j+8] = C_tile[4+j+16] - C_tile[8+j+16] - C_tile[12+j+16];          
  }

  int idx = out_thread[round][threadIdx.y%4] + threadIdx.y/4 + offset;
  c_tensor += idx*c_glb_offset;
  int x, x1;

  for(int i=0; i<2; i++){
    x = i*4;
    //x1 = i*(in_n*(tiles_dim-1) + in_n/2)*2;
    x1 = i*(in_n*(tiles_dim-(out_w%2)) + (out_w%2)*in_n/2)*2;
    if(mask&(1<<(i*2))){
      pOutputs[c_tensor+ x1] = At[x] + At[x+1] + At[x+2];
      pOutputs[c_tensor+2*c_glb_offset+x1] = At[x+8] + At[x+1+8] + At[x+2+8];
    }

    if(mask&(1<<(i*2+1))){
      pOutputs[c_tensor+x1+in_n] = At[x+1] - At[x+2] - At[x+3];
      pOutputs[c_tensor+2*c_glb_offset+x1+in_n] = At[x+1+8] - At[x+2+8] - At[x+3+8];          
    }
  }

}

__device__ __inline__ void store_output_tile(float4 acumm_smem[][16], float *shared_mem, float *C, int out_h, int out_w, int tiles_dim, int in_n, float4 *input_frag_mem, float4* filter_frag_mem, int out_thread[][4], int access_s_out[][16], short mask){
                                      
  float4 *output_smem = (float4 *) shared_mem;
  float4 *accumulator = (float4 *) acumm_smem;  

  float *C_tile = (float*) input_frag_mem;
  float *At = (float*) filter_frag_mem;

  mask = 0x000F;
  if((blockIdx.y/tiles_dim)==(tiles_dim-1) && out_w%2) mask&=0x0003;
  if(!((blockIdx.y+1)%tiles_dim) && out_w%2)           mask&=0X0005;

  // output transpose step
  int t=0;
  int acumm1, acumm2;

  acumm1 = access_s_out[0][threadIdx.x%8 + (threadIdx.x/16)*8];
  acumm2 = access_s_out[1][threadIdx.x%8 + (threadIdx.x/16)*8]; 

  int offset = BN_p*4;
  int init = (threadIdx.y/4)*BN_p*16*4 + (threadIdx.y%4)*40 + threadIdx.x; 
  int acumm3 = threadIdx.y * BN_p;
  int acumm4 = BN_p*8*2;

  int idx  = acumm3;
  int idx2 = idx + BN_p*8;

  float* out = (float *) output_smem;

  int c_glb_offset = in_n*out_h*out_w;
  int c_tensor = blockIdx.z*in_n*out_h*out_w*BK + (blockIdx.y%tiles_dim)*in_n*2 + (blockIdx.y/tiles_dim)*in_n*out_w*2 + blockIdx.x*BN + threadIdx.x;

  //#pragma unroll                                  
  for(int round=0; round<4; round++){

    //transformation step    
    if ( ((!round || round==1) && (threadIdx.x&15)<8) || ((round==2 || round==3) && (threadIdx.x&15)>7) ){

        #pragma unroll   
        for(int i=0; i<4; i+=2){

            *( (float4*) (output_smem + idx+i*acumm4 + acumm1) )  = *(accumulator+t);            // k=0
            *( (float4*) (output_smem + idx+i*acumm4 + acumm2) )  = *(accumulator+t+1);
            *( (float4*) (output_smem + idx+(i+1)*acumm4 + acumm1) )  = *(accumulator+2+t);   // k=1
            *( (float4*) (output_smem + idx+(i+1)*acumm4 + acumm2) )  = *(accumulator+3+t);

            *( (float4*) (output_smem + idx2+i*acumm4 + acumm1) ) = *(accumulator+16+t);
            *( (float4*) (output_smem + idx2+i*acumm4 + acumm2) ) = *(accumulator+17+t);
            *( (float4*) (output_smem + idx2+(i+1)*acumm4 + acumm1) ) = *(accumulator+18+t);
            *( (float4*) (output_smem + idx2+(i+1)*acumm4 + acumm2) ) = *(accumulator+19+t);

            t+=4;
        }
    } 
    __syncthreads();
    
    for(int i=0; i<16; i++){
        C_tile[i] =    out[init + i*offset];
        C_tile[i+16] = out[init + 2*BN_p*16*4 + i*offset];
    }

    // transform output tiles
    transform_output_tile(C, C_tile, At, out_h, out_w, tiles_dim, round, in_n, 0, out_thread, mask, c_tensor, c_glb_offset);
                        

    __syncthreads();
  }
  
}

}
#endif     
