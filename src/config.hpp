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


#ifndef COMMON_INCLUDE_FILE
#define COMMON_INCLUDE_FILE

#define BC 8
#define BN 32
#define BK 64
///////////////////// For Non-Fused version
#define BC_GEMM 8
#define BN_GEMM 128
#define BK_GEMM 128
///////////////////// For Non-Fused version

#ifdef OPTSTS64_CMP
#define BN_p 128
#elif BASE
#define BN_p 40
#else 
#define BN_p 138
#endif

#define N 128 // values: 32,64,96,128
#define C_in 256 // values: 64,128,256,512
#define W 14 // values: 56,28,14,7

#define K 256 // values: 64,128,256,512
#define R 3 // values: 3

#define PAD_H 1
#define PAD_W 1
#define STR_H 1
#define STR_W 1
#define DIL_H 1
#define DIL_W 1

#define M 2 // values: 2

__constant__ int access_f_s[2][32];
__constant__ int access_s[2][32];
#ifndef BASE
__constant__ int access_s_out[32];
__constant__ int out_thread[2][4][4];
__constant__ int out_sgemm[32];
__constant__ int exhange[32];
#else 
__constant__ int access_s_out[2][16];
__constant__ int out_thread[4][4];
#endif


// access_f_s
const int aux[2][32] = {
                        {0,0,1,1,2,2,3,3,4,4,5,5,6,6,
                            7,7,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7},
                        {8,8,9,9,10,10,11,11,12,12,13,13,
                            14,14,15,15,8,8,9,9,10,10,11,11,12,12,
                            13,13,14,14,15,15}
                        };
// access_s
const int aux2[2][32] = {
                         {0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,2,
                             3,2,3,2,3,2,3,2,3,2,3,2,3,2,3},
                         {4,5,4,5,4,5,4,5,4,5,4,
                            5,4,5,4,5,6,7,6,7,6,7,6,7,
                            6,7,6,7,6,7,6,7}
                        };                

#ifndef BASE
// access_s_out
const int aux3[32] = { 
                        0,1,34,35,68,69,102,103,  // first quarter
                        8,9,42,43,76,77,110,111,  // second quarter  
                        2,3,36,37,70,71,104,105,  // third quarter
                        10,11,44,45,78,79,112,113 // fourth quarter          
                        };
// out_thread
const int aux4[2][4][4] = { {{0,4,8,12}, {2,6,10,14},
                            {32,36,40,44}, {34,38,42,46} },
                            {{16,20,24,28}, {18,22,26,30},
                            {48,52,56,60}, {50,54,58,62}}};
// out_sgemm
const int aux5[32] = { 0,1,8,9,16,17,24,25,
                       32,33,40,41,48,49,56,57,
                       2,3,10,11,18,19,26,27,
                       34,35,42,43,50,51,58,59
                        };
// exhange                        
const int aux6[32] = {
                        2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13,
                        18,19,16,17,22,23,20,21,26,27,24,25,30,31,28,29
                    };     
 
#else
const int aux3[2][16] = { 
                      {0,1,10,11,20,21,30,31, 2,3,12,13,22,23,32,33},
                      {4,5,14,15,24,25,34,35, 6,7,16,17,26,27,36,37}
                    };
const int aux4[4][4] = { {0,4,8,12}, {32,36,40,44}, {16,20,24,28}, {48,52,56,60} }; 
#endif

#endif
