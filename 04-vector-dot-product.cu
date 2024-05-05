#include <cstdlib>
#include <stdio.h>

#define THREADS_PER_BLOCK 1024

#define HANDLE_CUDA_ERROR(expression)                                                              \
    {                                                                                              \
        cudaError_t error = (expression);                                                          \
        if (error != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", (__FILE__), (__LINE__),                   \
                    cudaGetErrorString(error));                                                    \
            exit(1);                                                                               \
        }                                                                                          \
    }

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
    int blocks_num = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int sums_blocks_num = (blocks_num + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    float *vector_a = (float *)malloc(n * sizeof(float));
    float *vector_b = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        vector_a[i] = 2;
        vector_b[i] = 2;
    }

    float *sums = (float *)malloc(sums_blocks_num * sizeof(float));

    float *dev_vector_a, *dev_vector_b, *dev_block_result, *dev_sums;
    HANDLE_CUDA_ERROR(cudaMalloc(&dev_vector_a, n * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&dev_vector_b, n * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&dev_block_result, blocks_num * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&dev_sums, sums_blocks_num * sizeof(float)));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    HANDLE_CUDA_ERROR(
        cudaMemcpy(dev_vector_a, vector_a, n * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(
        cudaMemcpy(dev_vector_b, vector_b, n * sizeof(float), cudaMemcpyHostToDevice));
    vectorDotProduct<<<blocks_num, THREADS_PER_BLOCK>>>(dev_vector_a, dev_vector_b,
                                                        dev_block_result, n);
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

    vectorSum<<<sums_blocks_num, THREADS_PER_BLOCK>>>(dev_block_result, dev_sums, blocks_num);
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    HANDLE_CUDA_ERROR(cudaMemcpy(sums, dev_sums, sums_blocks_num * sizeof(float), cudaMemcpyDeviceToHost));

    float result = 0.0f;
    for (int i = 0; i < sums_blocks_num; i++) {
        result += sums[i];
    }

    cudaEventRecord(end);

    printf("Dot product: %f\n", result);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    printf("Computed in %.3f milliseconds\n", milliseconds);

    cudaEventDestroy(end);
    cudaEventDestroy(start);

    return 0;
}
