#include <stdio.h>

#define THREADS_PER_BLOCK 1024

/*
Calculate the dot product of two vectors
*/
__global__ void vectorDotProduct(float *vector_a, float *vector_b, float *block_result, int n) {
    __shared__ float partial_sums[THREADS_PER_BLOCK];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        partial_sums[threadIdx.x] = vector_a[index] * vector_b[index];
    } else {
        partial_sums[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < THREADS_PER_BLOCK; i++) {
            sum += partial_sums[i];
        }
        block_result[blockIdx.x] = sum;
    }
}

// Interestingly enough this is slower than naive implementation above:
// around 52ms for 50 million elements vs 50ms for naive implementation
__global__ void vectorDotProductHierarchical(float *vector_a, float *vector_b, float *block_result,
                                             int n) {
    __shared__ float partial_sums[THREADS_PER_BLOCK];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        partial_sums[threadIdx.x] = vector_a[index] * vector_b[index];
    } else {
        partial_sums[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_result[blockIdx.x] = partial_sums[0];
    }
}

__global__ void vectorSum(float *a, float *block_result, int n) {
    __shared__ float partial_sums[THREADS_PER_BLOCK];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        partial_sums[threadIdx.x] = a[index];
    } else {
        partial_sums[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < THREADS_PER_BLOCK; i++) {
            sum += partial_sums[i];
        }
        block_result[blockIdx.x] = sum;
    }
}

int main() {
    int n = 200000000;

    float *vector_a, *vector_b, *block_result, *sums;
    cudaMallocManaged(&vector_a, n * sizeof(float));
    cudaMallocManaged(&vector_b, n * sizeof(float));
    int blocks_num = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaMallocManaged(&block_result, blocks_num * sizeof(float));
    int sums_blocks_num = (blocks_num + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaMallocManaged(&sums, sums_blocks_num * sizeof(float));

    for (int i = 0; i < n; i++) {
        vector_a[i] = 2;
        vector_b[i] = 2;
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    vectorDotProduct<<<blocks_num, THREADS_PER_BLOCK>>>(vector_a, vector_b, block_result, n);
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    }

    vectorSum<<<sums_blocks_num, THREADS_PER_BLOCK>>>(block_result, sums, blocks_num);
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    }

    float result = 0.0f;
    for (int i = 0; i < sums_blocks_num; i++) {
        result += sums[i];
    }

    cudaEventRecord(end);

    printf("Dot product: %f\n", result);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    printf("Computed in %.3f milliseconds\n", milliseconds);
    return 0;
}
