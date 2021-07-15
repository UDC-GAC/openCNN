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


__device__  __forceinline__ void outer_product(float4* input_frag, float4* filter_frag, float4 accumulator[][16]){
    accumulator[0][0].x += input_frag[0].x*filter_frag[0].x;
    accumulator[0][0].z += input_frag[0].y*filter_frag[0].x;
    accumulator[0][0].y += input_frag[0].z*filter_frag[0].x;
    accumulator[0][0].w += input_frag[0].w*filter_frag[0].x;
    
    accumulator[0][1].x += input_frag[1].x*filter_frag[0].x;
    accumulator[0][1].z += input_frag[1].y*filter_frag[0].x;
    accumulator[0][1].y += input_frag[1].z*filter_frag[0].x;                                     
    accumulator[0][1].w += input_frag[1].w*filter_frag[0].x;
  
    accumulator[0][2].x += input_frag[0].x*filter_frag[0].y;
    accumulator[0][2].z += input_frag[0].y*filter_frag[0].y;                                
    accumulator[0][2].y += input_frag[0].z*filter_frag[0].y;                                
    accumulator[0][2].w += input_frag[0].w*filter_frag[0].y;                                  
  
    accumulator[0][3].x += input_frag[1].x*filter_frag[0].y;
    accumulator[0][3].z += input_frag[1].y*filter_frag[0].y;                                 
    accumulator[0][3].y += input_frag[1].z*filter_frag[0].y;                                 
    accumulator[0][3].w += input_frag[1].w*filter_frag[0].y;
                                      
    accumulator[0][4].x += input_frag[0].x*filter_frag[0].z;
    accumulator[0][4].z += input_frag[0].y*filter_frag[0].z;                                     
    accumulator[0][4].y += input_frag[0].z*filter_frag[0].z;                                     
    accumulator[0][4].w += input_frag[0].w*filter_frag[0].z;
                                        
    accumulator[0][5].x += input_frag[1].x*filter_frag[0].z;
    accumulator[0][5].z += input_frag[1].y*filter_frag[0].z;                                     
    accumulator[0][5].y += input_frag[1].z*filter_frag[0].z;                                     
    accumulator[0][5].w += input_frag[1].w*filter_frag[0].z;
  
    accumulator[0][6].x += input_frag[0].x*filter_frag[0].w;
    accumulator[0][6].z += input_frag[0].y*filter_frag[0].w;                                   
    accumulator[0][6].y += input_frag[0].z*filter_frag[0].w;                                   
    accumulator[0][6].w += input_frag[0].w*filter_frag[0].w;
                                        
    accumulator[0][7].x += input_frag[1].x*filter_frag[0].w;
    accumulator[0][7].z += input_frag[1].y*filter_frag[0].w;                                    
    accumulator[0][7].y += input_frag[1].z*filter_frag[0].w;                                    
    accumulator[0][7].w += input_frag[1].w*filter_frag[0].w;
  
    //
    accumulator[0][8].x += input_frag[0].x*filter_frag[1].x;
    accumulator[0][8].z += input_frag[0].y*filter_frag[1].x;
    accumulator[0][8].y += input_frag[0].z*filter_frag[1].x;
    accumulator[0][8].w += input_frag[0].w*filter_frag[1].x;
    
    accumulator[0][9].x += input_frag[1].x*filter_frag[1].x;
    accumulator[0][9].z += input_frag[1].y*filter_frag[1].x;
    accumulator[0][9].y += input_frag[1].z*filter_frag[1].x;                                     
    accumulator[0][9].w += input_frag[1].w*filter_frag[1].x;
  
    accumulator[0][10].x += input_frag[0].x*filter_frag[1].y;
    accumulator[0][10].z += input_frag[0].y*filter_frag[1].y;                                
    accumulator[0][10].y += input_frag[0].z*filter_frag[1].y;                                
    accumulator[0][10].w += input_frag[0].w*filter_frag[1].y;                                  
  
    accumulator[0][11].x += input_frag[1].x*filter_frag[1].y;
    accumulator[0][11].z += input_frag[1].y*filter_frag[1].y;                                 
    accumulator[0][11].y += input_frag[1].z*filter_frag[1].y;                                 
    accumulator[0][11].w += input_frag[1].w*filter_frag[1].y;
                                      
    accumulator[0][12].x += input_frag[0].x*filter_frag[1].z;
    accumulator[0][12].z += input_frag[0].y*filter_frag[1].z;                                     
    accumulator[0][12].y += input_frag[0].z*filter_frag[1].z;                                     
    accumulator[0][12].w += input_frag[0].w*filter_frag[1].z;
                                        
    accumulator[0][13].x += input_frag[1].x*filter_frag[1].z;
    accumulator[0][13].z += input_frag[1].y*filter_frag[1].z;                                     
    accumulator[0][13].y += input_frag[1].z*filter_frag[1].z;                                     
    accumulator[0][13].w += input_frag[1].w*filter_frag[1].z;
  
    accumulator[0][14].x += input_frag[0].x*filter_frag[1].w;
    accumulator[0][14].z += input_frag[0].y*filter_frag[1].w;                                   
    accumulator[0][14].y += input_frag[0].z*filter_frag[1].w;                                   
    accumulator[0][14].w += input_frag[0].w*filter_frag[1].w;
                                        
    accumulator[0][15].x += input_frag[1].x*filter_frag[1].w;
    accumulator[0][15].z += input_frag[1].y*filter_frag[1].w;                                    
    accumulator[0][15].y += input_frag[1].z*filter_frag[1].w;                                    
    accumulator[0][15].w += input_frag[1].w*filter_frag[1].w;
  
    //////
    accumulator[1][0].x += input_frag[2].x*filter_frag[2].x;
    accumulator[1][0].z += input_frag[2].y*filter_frag[2].x;
    accumulator[1][0].y += input_frag[2].z*filter_frag[2].x;
    accumulator[1][0].w += input_frag[2].w*filter_frag[2].x;
    
    accumulator[1][1].x += input_frag[3].x*filter_frag[2].x;
    accumulator[1][1].z += input_frag[3].y*filter_frag[2].x;
    accumulator[1][1].y += input_frag[3].z*filter_frag[2].x;                                     
    accumulator[1][1].w += input_frag[3].w*filter_frag[2].x;
  
    accumulator[1][2].x += input_frag[2].x*filter_frag[2].y;
    accumulator[1][2].z += input_frag[2].y*filter_frag[2].y;                                
    accumulator[1][2].y += input_frag[2].z*filter_frag[2].y;                                
    accumulator[1][2].w += input_frag[2].w*filter_frag[2].y;                                  
  
    accumulator[1][3].x += input_frag[3].x*filter_frag[2].y;
    accumulator[1][3].z += input_frag[3].y*filter_frag[2].y;                                 
    accumulator[1][3].y += input_frag[3].z*filter_frag[2].y;                                 
    accumulator[1][3].w += input_frag[3].w*filter_frag[2].y;
                                      
    accumulator[1][4].x += input_frag[2].x*filter_frag[2].z;
    accumulator[1][4].z += input_frag[2].y*filter_frag[2].z;                                     
    accumulator[1][4].y += input_frag[2].z*filter_frag[2].z;                                     
    accumulator[1][4].w += input_frag[2].w*filter_frag[2].z;
                                        
    accumulator[1][5].x += input_frag[3].x*filter_frag[2].z;
    accumulator[1][5].z += input_frag[3].y*filter_frag[2].z;                                     
    accumulator[1][5].y += input_frag[3].z*filter_frag[2].z;                                     
    accumulator[1][5].w += input_frag[3].w*filter_frag[2].z;
  
    accumulator[1][6].x += input_frag[2].x*filter_frag[2].w;
    accumulator[1][6].z += input_frag[2].y*filter_frag[2].w;                                   
    accumulator[1][6].y += input_frag[2].z*filter_frag[2].w;                                   
    accumulator[1][6].w += input_frag[2].w*filter_frag[2].w;
                                        
    accumulator[1][7].x += input_frag[3].x*filter_frag[2].w;
    accumulator[1][7].z += input_frag[3].y*filter_frag[2].w;                                    
    accumulator[1][7].y += input_frag[3].z*filter_frag[2].w;                                    
    accumulator[1][7].w += input_frag[3].w*filter_frag[2].w;
  
    //
    accumulator[1][8].x += input_frag[2].x*filter_frag[3].x;
    accumulator[1][8].z += input_frag[2].y*filter_frag[3].x;
    accumulator[1][8].y += input_frag[2].z*filter_frag[3].x;
    accumulator[1][8].w += input_frag[2].w*filter_frag[3].x;
    
    accumulator[1][9].x += input_frag[3].x*filter_frag[3].x;
    accumulator[1][9].z += input_frag[3].y*filter_frag[3].x;
    accumulator[1][9].y += input_frag[3].z*filter_frag[3].x;                                     
    accumulator[1][9].w += input_frag[3].w*filter_frag[3].x;
  
    accumulator[1][10].x += input_frag[2].x*filter_frag[3].y;
    accumulator[1][10].z += input_frag[2].y*filter_frag[3].y;                                
    accumulator[1][10].y += input_frag[2].z*filter_frag[3].y;                                
    accumulator[1][10].w += input_frag[2].w*filter_frag[3].y;                                  
  
    accumulator[1][11].x += input_frag[3].x*filter_frag[3].y;
    accumulator[1][11].z += input_frag[3].y*filter_frag[3].y;                                 
    accumulator[1][11].y += input_frag[3].z*filter_frag[3].y;                                 
    accumulator[1][11].w += input_frag[3].w*filter_frag[3].y;
                                      
    accumulator[1][12].x += input_frag[2].x*filter_frag[3].z;
    accumulator[1][12].z += input_frag[2].y*filter_frag[3].z;                                     
    accumulator[1][12].y += input_frag[2].z*filter_frag[3].z;                                     
    accumulator[1][12].w += input_frag[2].w*filter_frag[3].z;
                                        
    accumulator[1][13].x += input_frag[3].x*filter_frag[3].z;
    accumulator[1][13].z += input_frag[3].y*filter_frag[3].z;                                     
    accumulator[1][13].y += input_frag[3].z*filter_frag[3].z;                                     
    accumulator[1][13].w += input_frag[3].w*filter_frag[3].z;
  
    accumulator[1][14].x += input_frag[2].x*filter_frag[3].w;
    accumulator[1][14].z += input_frag[2].y*filter_frag[3].w;                                   
    accumulator[1][14].y += input_frag[2].z*filter_frag[3].w;                                   
    accumulator[1][14].w += input_frag[2].w*filter_frag[3].w;
                                        
    accumulator[1][15].x += input_frag[3].x*filter_frag[3].w;
    accumulator[1][15].z += input_frag[3].y*filter_frag[3].w;                                    
    accumulator[1][15].y += input_frag[3].z*filter_frag[3].w;                                    
    accumulator[1][15].w += input_frag[3].w*filter_frag[3].w;
  }