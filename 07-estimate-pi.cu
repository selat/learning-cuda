#include <curand_kernel.h>
#include <stdio.h>

/*
Estimate Pi using Monte Carlo method, counting how many points fall inside a circle

Example output:
Trials: 10240000000000, Count: 8042477486365
Pi is roughly 3.141592768
*/

__global__ void kernelEstimatePi(int n, unsigned long long *count) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(42, id, 0, &state);

    int local_count = 0;
    for (int i = 0; i < n; ++i) {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x * x + y * y <= 1) {
            ++local_count;
        }
    }

    atomicAdd(count, local_count);
}

int main() {
    int n = 1000 * 1000;

    unsigned long long *count;
    cudaMallocManaged(&count, sizeof(int));
    *count = 0;

    const int blocks_num = 10000;
    const int threads_num = 1024;
    unsigned long long trials = (unsigned long long)blocks_num * threads_num * n;
    kernelEstimatePi<<<blocks_num, threads_num>>>(n, count);
    cudaError_t cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cuda_error));
        return 1;
    }

    printf("Trials: %lld, Count: %lld\n", trials, *count);
    double pi = 4.0 * *count / trials;
    printf("Pi is roughly %.9f\n", pi);

    cudaFree(count);

    return 0;
}
