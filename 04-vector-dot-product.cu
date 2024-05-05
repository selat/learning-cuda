#include <cstdlib>
#include <stdio.h>
#include <sys/time.h>

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

Example output:
N = 200000000
Dot product: 800000000.000000
Computed in 7.827 milliseconds
Dot product: 800000000.000000
Hierarchical computed in 1.946 milliseconds
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

__global__ void vectorDotProductHierarchical(float *vector_a, float *vector_b, float *block_result,
                                             int n) {
    __shared__ float partial_sums[THREADS_PER_BLOCK];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float current_sum = 0.0f;
    while (index < n) {
        current_sum += vector_a[index] * vector_b[index];
        index += gridDim.x * blockDim.x;
    }

    partial_sums[threadIdx.x] = current_sum;
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

float getElapsedMilliseconds(struct timeval *start, struct timeval *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 + (end->tv_usec - start->tv_usec) / 1000.0;
}

int main() {
    int n = 200000000;
    int blocks_num = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int sums_blocks_num = (blocks_num + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("N = %d\n", n);

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
    HANDLE_CUDA_ERROR(
        cudaMemcpy(dev_vector_a, vector_a, n * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(
        cudaMemcpy(dev_vector_b, vector_b, n * sizeof(float), cudaMemcpyHostToDevice));

    struct timeval cpu_start, cpu_end;
    gettimeofday(&cpu_start, NULL);
    vectorDotProduct<<<blocks_num, THREADS_PER_BLOCK>>>(dev_vector_a, dev_vector_b,
                                                        dev_block_result, n);
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

    vectorSum<<<sums_blocks_num, THREADS_PER_BLOCK>>>(dev_block_result, dev_sums, blocks_num);
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    HANDLE_CUDA_ERROR(cudaMemcpy(sums, dev_sums, sums_blocks_num * sizeof(float),
    cudaMemcpyDeviceToHost));

    float result = 0.0f;
    for (int i = 0; i < sums_blocks_num; i++) {
        result += sums[i];
    }

    gettimeofday(&cpu_end, NULL);

    printf("Dot product: %f\n", result);

    printf("Computed in %.3f milliseconds\n", getElapsedMilliseconds(&cpu_start, &cpu_end));

    gettimeofday(&cpu_start, NULL);
    vectorDotProductHierarchical<<<sums_blocks_num, THREADS_PER_BLOCK>>>(dev_vector_a, dev_vector_b,
                                                                         dev_sums, n);
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    HANDLE_CUDA_ERROR(
        cudaMemcpy(sums, dev_sums, sums_blocks_num * sizeof(float), cudaMemcpyDeviceToHost));
    result = 0.0f;
    for (int i = 0; i < sums_blocks_num; i++) {
        result += sums[i];
    }

    gettimeofday(&cpu_end, NULL);

    printf("Dot product: %f\n", result);
    printf("Hierarchical computed in %.3f milliseconds\n",
           getElapsedMilliseconds(&cpu_start, &cpu_end));

    cudaFree(dev_vector_a);
    cudaFree(dev_vector_b);
    cudaFree(dev_block_result);
    cudaFree(dev_sums);

    free(vector_a);
    free(vector_b);
    free(sums);

    return 0;
}
