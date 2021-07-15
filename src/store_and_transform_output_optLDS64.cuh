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


#include "config.hpp"

#ifndef _OUTPUT_KERNEL_OPT3_
#define _OUTPUT_KERNEL_OPT3_
extern "C"
{

__device__ void  __inline__ transform_output_tile(float2 *pOutputs, float2 *C_tile, float2 *At, int tiles_dim, int round, int in_n, int c_tensor, int c_glb_offset, short mask, int out_w)
{
  c_tensor += ( (round/2)*32 + (round%2)*2 )*c_glb_offset/2;  
  int x, x1;

  for(int j=0; j<4; j++){
      At[j].x = C_tile[j].x + C_tile[4+j].x + C_tile[8+j].x;
      At[j].y = C_tile[j].y + C_tile[4+j].y + C_tile[8+j].y;
    
      At[4+j].x = C_tile[4+j].x - C_tile[8+j].x - C_tile[12+j].x;
      At[4+j].y = C_tile[4+j].y - C_tile[8+j].y - C_tile[12+j].y;      
  }

  for(int i=0; i<2; i++){
    x = i*4;
    x1 = i*(in_n*(tiles_dim-(out_w%2)) + (out_w%2)*in_n/2);
    if(mask&(1<<(i*2))){

      pOutputs[x1 + c_tensor].x = At[x].x + At[x+1].x + At[x+2].x;
      pOutputs[x1 + c_tensor].y = At[x].y + At[x+1].y + At[x+2].y;
    }
    if(mask&(1<<(i*2+1))){
      pOutputs[x1 + in_n/2 + c_tensor].x = At[x+1].x - At[x+2].x - At[x+3].x;
      pOutputs[x1 + in_n/2 + c_tensor].y = At[x+1].y - At[x+2].y - At[x+3].y;
    }
  }

}

__device__ __inline__ void store_output_tile(float4 acumm_smem[][16], float  *shared_mem, int Inx, int Iny,
                                  float *C, int TileX, int TileY, int TileZ, int out_h, int out_w, 
                                  int tiles_dim, int in_n, float4 *input_frag_mem, float4* filter_frag_mem, short mask){
  
  float2 *output_smem = (float2 *) shared_mem;
  float2 *accumulator = (float2 *) acumm_smem;
  float2 *C_out = (float2*)C;

  float2 *C_tile = (float2*) input_frag_mem;
  float2 *At = (float2*) filter_frag_mem;

  mask = 0x000F;
  if((blockIdx.y/tiles_dim)==(tiles_dim-1) && out_w%2) mask&=0x0003;
  if(!((blockIdx.y+1)%tiles_dim) && out_w%2)           mask&=0X0005;
  
  // output transpose step
  int t=0;
  int acumm1, acumm2;
  // For transposing
  //acumm1 = access_s_out[Inx]; //* 4
  acumm1 = ((Inx%8)/2)*34 + Inx%2 + (Inx/16)*2 + ((Inx/8)%2)*8;
  acumm2 = acumm1+4;
                                    
  int acumm4 = BN_p*8*2 ; //*4
  int idx  = Iny * BN_p;
  int idx2 = idx + BN_p*8; //(BN_p*2 *8)/2

  // For transformating
  int offset = BN_p; //*2/2
  int init = (Iny/4)*BN_p*16 + (Iny%4)*(32+2);
  init += (Inx/16)*8 + ((Inx/8)%2)*16 + (Inx%8); //40=(8+2)*4, 4 blocks/buffer


  int c_glb_offset = in_n*out_h*out_w;                    
  int c_tensor = TileZ*c_glb_offset*BK + (TileY%tiles_dim)*in_n*2 + (TileY/tiles_dim)*in_n*out_w*2 + TileX*BN + ((Inx/8)%2)*2 + (Inx%8)*2*2 + ((Inx/16)*16 + (Iny%4)*4 + Iny/4)*c_glb_offset;
  c_tensor/=2; 

  #pragma unroll                                  
  for(int round=0; round<4; round++){

    *( (float2*) (output_smem + idx + acumm1) )  = *(accumulator+t);
    *( (float2*) (output_smem + idx + acumm1 + 16) )  = *(accumulator+t+1); // float 4, t
    *( (float2*) (output_smem + idx + acumm2) )  = *(accumulator+t+2);
    *( (float2*) (output_smem + idx + acumm2 + 16) )  = *(accumulator+t+3); // float 4, t+1

    *( (float2*) (output_smem + idx2 + acumm1) ) = *(accumulator+t+32);
    *( (float2*) (output_smem + idx2 + acumm1 + 16) ) = *(accumulator+t+33); // float 4, t+16
    *( (float2*) (output_smem + idx2 + acumm2) ) = *(accumulator+t+34);
    *( (float2*) (output_smem + idx2 + acumm2 + 16) ) = *(accumulator+t+35); // float 4, t+17

    *( (float2*) (output_smem + idx + acumm4 + acumm1) )  = *(accumulator+t+4); 
    *( (float2*) (output_smem + idx + acumm4 + acumm1 + 16) )  = *(accumulator+t+5); // float 4, t+2
    *( (float2*) (output_smem + idx + acumm4 + acumm2) )  = *(accumulator+t+6);
    *( (float2*) (output_smem + idx + acumm4 + acumm2 + 16) )  = *(accumulator+t+7); // float 4, t+3

    *( (float2*) (output_smem + idx2 + acumm4 + acumm1) ) = *(accumulator+t+36);
    *( (float2*) (output_smem + idx2 + acumm4 + acumm1 + 16) ) = *(accumulator+t+37); // float 4, t+18
    *( (float2*) (output_smem + idx2 + acumm4 + acumm2) ) = *(accumulator+t+38);
    *( (float2*) (output_smem + idx2 + acumm4 + acumm2 + 16) ) = *(accumulator+t+39); // float 4, t+19

    t+=8;

    __syncthreads();


    for(int i=0; i<16; i++){
      C_tile[i].x = output_smem[i*offset + init].x; //16*4
      C_tile[i].y = output_smem[i*offset + init].y; //16*4
    }

    // transform output tiles
    transform_output_tile(C_out, C_tile, At, tiles_dim, round, in_n, c_tensor, c_glb_offset, mask , out_w);

    __syncthreads();
    
  }
}

}
#endif     