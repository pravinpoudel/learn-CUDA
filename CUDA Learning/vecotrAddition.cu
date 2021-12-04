#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<assert.h>
#include<cmath>

__global__ void multiply_kernel(int * A, int * B, int* C, int n) {

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < n) {
		C[tid] = A[tid] + B[tid];
	}
}

void error_check(int * A, int* B, int* C, int n) {
	for (size_t i = 0; i < n; i++)
	{
		assert(C[i] == A[i] + B[i]);
	}
}


void arrayInit(int* array, int n) {
	for (int i = 0; i < n; i++) {
		array[i] = rand() % 10000 / 1000.0f;
	}
}

int main() {
	
	int n = 1 << 16;
	//host memory
	int* h_A = new int[n];
	int* h_B = new int[n];
	int* h_C = new int[n];

	//device memory
	int* d_A, * d_B, *d_C;

	cudaMalloc(&d_A, n);
	cudaMalloc(&d_B, n);
	cudaMalloc(&d_C, n);
	
	arrayInit(h_A, n);
	arrayInit(h_B, n);

	cudaMemcpy(d_A, h_A, sizeof(int)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizeof(int) * n, cudaMemcpyHostToDevice);

	//here put number of thread best according to the architecture of the CPU
	//but it is good to make size of multiple of 32 because these havr to translate it to warps which are of size 32
	int NUM_THREADS = 256;
	int NUM_BLOCKS = int(ceil(n / NUM_THREADS));

	//launch kernel on different stream 
	multiply_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_A, d_B, d_C, n);

	cudaMemcpy(h_C, d_C, sizeof(int) * n, cudaMemcpyDeviceToHost);

	error_check(h_A, h_B, h_C, n);
	delete[] h_A;
	delete[] h_B;

	return 0;
}