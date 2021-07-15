#include <cuda.h>
#include <stdio.h>
#include <string.h>

#define ITERS 32768

char* concat(const char *s1, const char *s2)
{
    char *result = (char*)malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
    // in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

#define CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

void initializeEvents(cudaEvent_t *start, cudaEvent_t *stop){
	CUDA_SAFE_CALL( cudaEventCreate(start) );
	CUDA_SAFE_CALL( cudaEventCreate(stop) );
	CUDA_SAFE_CALL( cudaEventRecord(*start, 0) );
}

float finalizeEvents(cudaEvent_t start, cudaEvent_t stop){
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaEventRecord(stop, 0) );
	CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
	float kernel_time;
	CUDA_SAFE_CALL( cudaEventElapsedTime(&kernel_time, start, stop) );
	CUDA_SAFE_CALL( cudaEventDestroy(start) );
	CUDA_SAFE_CALL( cudaEventDestroy(stop) );
	return kernel_time;
}

template<typename T>
void run(char * name, T scal, int type_size, int threads){
	char * file_name = concat(name, ".cubin");

	int *output;
	cudaMalloc((void**)&output, sizeof(int)*32);
	cudaMemset(output, 0, 32*sizeof(int));

	CUmodule module;
	CUfunction kernel;

	cuModuleLoad(&module, file_name);
	cuModuleGetFunction(&kernel, module, "kern");

	int blk_size = 32;
	int total_blks = 1;//size/blk_size;	
	int sh_mem_size = blk_size*sizeof(float)*type_size;
	void * args[2] = {&scal, &output};

	//cudaEvent_t start, stop;
	//initializeEvents(&start, &stop);
	cuLaunchKernel(kernel, total_blks, 1, 1,
			blk_size, 1, 1,
			sh_mem_size, 0, args, 0);
	//float krn_time_shmem_32b = finalizeEvents(start, stop);

	int *output_h = (int*)malloc(sizeof(int)*32);	

	cudaMemcpy(output_h, output, sizeof(int)*32, cudaMemcpyDeviceToHost);	

	/*for(int i=0; i<32; i++){
		printf("%d ", output_h[i]);
	}printf("\n");*/

	printf("%s took %d clocks \n", name, output_h[0]);
	double clocks_instr = (float)output_h[0]/(128.0*128.0); // wokload of a thread
	printf("Each instruction takes %.2f clocks.\n", clocks_instr);
	printf("Throughput %.2f bytes/cycle.\n\n", ((double)threads*128*128*type_size*4)/output_h[0]); // Size of information stores divided by the number of threads of the latest thread

	cudaFree(output);
	free(output_h);
}

int main(){
	float scal = 4;
	run("sts32", scal, 1, 32);
	printf("\n");
	float2 scal2;
	scal2.x = 4; scal2.y = 4;
	run("sts64", scal2, 2, 32);
	printf("\n");
	float4 scal4;
	scal4.x = 4; scal4.y = 4;
	scal4.z = 4; scal4.w = 4;
	// No thread-divergence
	run("sts128_0", scal4, 4, 32);

	printf("\n");
	// Only half of the threads store data
	run("sts128", scal4, 4, 16);

	/* run2_aux("sts64_2bank_conflict");
	printf("\n");
	run2_aux("sts64_broadcast");
	printf("\n");
	run2_aux("sts64_opt3");
	printf("\n"); */

	return 0;
}
