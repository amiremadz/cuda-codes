
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// Each block takes care of 1024 elements
__global__
void global_reduce(float* gpu_out, float* gpu_in){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tidx = threadIdx.x;

	int s = blockDim.x/2;
	
	while(s > 0 ){
		if(tidx < s){
			gpu_in[tid] += gpu_in[tid + s];
		}
		s >>= 1;
		__syncthreads();
	}

    // only thread 0 writes result for this block back to global mem
	if(tidx == 0){
		gpu_out[blockIdx.x] = gpu_in[tid];
	}
}

__global__
void shmem_reduce(float* gpu_out, const float* gpu_in){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tidx = threadIdx.x;

	const int size = 1024;

	__shared__
	float smem[size];

	if(tidx < size){
		smem[tidx] = gpu_in[tid];
	}
	__syncthreads();

	int s = size/2;

	while(s > 0 ){
		if(tidx < s){
			smem[tidx] += smem[tidx + s];
		}
		s >>= 1;
		__syncthreads();
	}

    // only thread 0 writes result for this block back to global mem
	if(tidx == 0){
		gpu_out[blockIdx.x] = smem[tidx];
	}
}

void reduce(float *d_out, float *d_intermediate, float * d_in, 
            int size, bool usesSharedMemory){
	const int maxThreadsPerBlock = 1024;
	int threads = maxThreadsPerBlock;
	int blocks = size / maxThreadsPerBlock;

	if(usesSharedMemory){
		shmem_reduce<<<blocks, threads>>>(d_intermediate, d_in);
	}
	else{
		global_reduce<<<blocks, threads>>>(d_intermediate, d_in);
	}

	threads = maxThreadsPerBlock;
	blocks = 1;

	if(usesSharedMemory){
		shmem_reduce<<<blocks, threads>>>(d_out, d_intermediate);
	}
	else{
		global_reduce<<<blocks, threads>>>(d_out, d_intermediate);
	}
}

int main(){
	int dev = 0;
	cudaSetDevice(dev);

 	const int ARRAY_SIZE = 1 << 20;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
    float h_in[ARRAY_SIZE];
    float sum = 0.0f;
    for(int i = 0; i < ARRAY_SIZE; i++) {
        // generate random float in [-1.0f, 1.0f]
        h_in[i] = -1.0f + (float)random()/((float)RAND_MAX/2.0f);
        sum += h_in[i];
    }

	printf("sum using serial reduce: %f\n", sum);
    
    // declare GPU memory pointers
    float * d_in, * d_intermediate, * d_out;

    // allocate GPU memory
    HANDLE_ERROR(cudaMalloc((void **) &d_in, ARRAY_BYTES));
    HANDLE_ERROR(cudaMalloc((void **) &d_intermediate, ARRAY_BYTES)); // overallocated
    HANDLE_ERROR(cudaMalloc((void **) &d_out, sizeof(float)));

    // transfer the input array to the GPU
    HANDLE_ERROR(cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice)); 

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

 	printf("Running global mamory reduce\n");
    cudaEventRecord(start, 0);
    const int itt = 100;
    for (int i = 0; i < itt; i++){
    	HANDLE_ERROR(cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice)); 
        reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, false);
    }
    cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);    
    elapsedTime /= (float)itt;      // 100 trials

    // copy back the sum from GPU
    float h_out;
    HANDLE_ERROR(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    printf("sum using global memory kernel: %f\n", h_out);
    printf("average time elapsed using global memory kernel: %f\n", elapsedTime);

 	printf("Running shared mamory reduce\n");
   	HANDLE_ERROR(cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice)); 

    cudaEventRecord(start, 0);
    for (int i = 0; i < itt; i++){
        reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, true);
    }
    cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);    
    elapsedTime /= (float)itt;      // 100 trials

    // copy back the sum from GPU
    HANDLE_ERROR(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

	printf("sum using shared memory kernel: %f\n", h_out);
    printf("average time elapsed using shared memory kernel: %f\n", elapsedTime);

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);

	return 0;
}
