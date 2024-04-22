#include <stdio.h>
#include <sys/time.h>

/*
Find all prime numbers from 1 to n

Example output for n=1_000_000_000:
SieveOfEratosthenes computed in 8952.741ms, found 50847534 primes from 1 to 1000000000
Sqrt computed in 37806.473ms, found 50847534 primes from 1 to 1000000000
Approximate number of primes can be approximated by n / log(n)
*/

__global__ void kernelFindPrimes(int n, int* primes, int* primes_count) {
    int number = blockIdx.x * blockDim.x + threadIdx.x;

    if (number < 2) {
        return;
    }

    int is_prime = 1;
    for (int i = 2; i * i <= number; ++i) {
        if (number % i == 0) {
            is_prime = 0;
            break;
        }
    }
    if (is_prime) {
        int i = atomicAdd(primes_count, 1);
        primes[i] = number;
    }
}

__global__ void kernelSieveOfEratosthenes(int n, bool *is_prime) {
    int number = blockIdx.x * blockDim.x + threadIdx.x;

    // It's important to convert number to long long to avoid overflow
    if (number < 2 || (long long)number * number > n) {
        return;
    }

    if (is_prime[number]) {
        for (int i = number * number; i <= n; i += number) {
            is_prime[i] = false;
        }
    }
}

float getElapsedMilliseconds(struct timeval* start, struct timeval* end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 + (end->tv_usec - start->tv_usec) / 1000.0;
}

int main() {
    int n = 1000 * 1000 * 1000;

    int* primes;
    int* primes_count;
    bool* is_prime;
    cudaMallocManaged(&primes, n * sizeof(int));
    cudaMallocManaged(&primes_count, sizeof(int));
    *primes_count = 0;
    cudaMallocManaged(&is_prime, (n + 1) * sizeof(bool));
    is_prime[0] = false;
    is_prime[1] = false;
    for (int i = 2; i <= n; ++i) {
        is_prime[i] = true;
    }

    cudaEvent_t cuda_start, cuda_end;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_end);

    cudaEventRecord(cuda_start);
    kernelSieveOfEratosthenes<<<n, 1>>>(n, is_prime);
    cudaEventRecord(cuda_end);
    cudaError_t cudaError = cudaDeviceSynchronize();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        if (is_prime[i]) {
            ++(*primes_count);
        }
    }

    float gpu_milliseconds = 0;
    cudaEventElapsedTime(&gpu_milliseconds, cuda_start, cuda_end);

    printf("SieveOfEratosthenes computed in %.3fms, found %d primes from 1 to %d\n", gpu_milliseconds, *primes_count, n);

    *primes_count = 0;
    cudaEventRecord(cuda_start);
    kernelFindPrimes<<<n / 64, 64>>>(n, primes, primes_count);
    cudaEventRecord(cuda_end);
    cudaError = cudaDeviceSynchronize();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
        return 1;
    }

    gpu_milliseconds = 0;
    cudaEventElapsedTime(&gpu_milliseconds, cuda_start, cuda_end);
    printf("Sqrt computed in %.3fms, found %d primes from 1 to %d\n", gpu_milliseconds, *primes_count, n);

    cudaFree(primes);
    cudaFree(primes_count);
    cudaFree(is_prime);

    return 0;
}
