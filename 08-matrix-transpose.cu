#include <stdio.h>


/*
Transpose a matrix defined in row-major order
*/

__global__ void kernelTranspose(float* matrix, float* output, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m) {
        output[j * n + i] = matrix[i * m + j];
    }
}

void printMatrix(float* matrix, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            printf("%.2f ", matrix[i * m + j]);
        }
        printf("\n");
    }
}

int main() {
    const int n = 4;
    const int m = 3;

    float* matrix, *output_matrix;
    cudaMallocManaged(&matrix, n * m * sizeof(float));
    cudaMallocManaged(&output_matrix, m * n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            matrix[i * m + j] = i * m + j;
        }
    }

    printMatrix(matrix, n, m);
    printf("\n");

    dim3 threads(2, 2);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
    kernelTranspose<<<blocks, threads>>>(matrix, output_matrix, n, m);
    cudaError_t cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cuda_error));
        return 1;
    }

    printMatrix(output_matrix, m, n);

    return 0;
}
