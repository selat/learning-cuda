#include <stdio.h>
#include <sys/time.h>

/*
Multiply two matrices
*/
__global__ void multiplyMatrices(float *matrix_a, float *matrix_b, float *matrix_c, int rows_a,
                                 int cols_a, int cols_b) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_a && col < cols_b) {
        float sum = 0.0f;
        for (int i = 0; i < cols_a; i++) {
            sum += matrix_a[row * cols_a + i] * matrix_b[i * cols_b + col];
        }
        matrix_c[row * cols_b + col] = sum;
    }
}

float *allocateGPUMatrix(int rows, int cols) {
    float *matrix;
    cudaError_t err = cudaMallocManaged(&matrix, rows * cols * sizeof(float));
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    return matrix;
}

float *allocateMatrix(int rows, int cols) { return (float *)malloc(rows * cols * sizeof(float)); }

void setMatrixIdentity(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

void setMatrixRandom(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

void printMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%*.1f ", 4, matrix[i * cols + j]);
        }
        puts("");
    }
}

void cpuMultiplyMatrices(float *matrix_a, float *matrix_b, float *matrix_c, int rows_a, int cols_a,
                         int cols_b) {
    for (int row = 0; row < rows_a; ++row) {
        for (int col = 0; col < cols_b; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < cols_a; ++i) {
                sum += matrix_a[row * cols_a + i] * matrix_b[i * cols_b + col];
            }
            matrix_c[row * cols_b + col] = sum;
        }
    }
}

void testGPUMultiplication(int n) {
    float *matrix_a = allocateGPUMatrix(n, n);
    float *matrix_b = allocateGPUMatrix(n, n);
    float *matrix_c = allocateGPUMatrix(n, n);

    // Interestingly, there's no performance difference between multiplying
    // identity matrices and random matrices
    setMatrixRandom(matrix_a, n, n);
    setMatrixRandom(matrix_b, n, n);

    cudaEvent_t multiplication_start, multiplication_end;
    cudaEventCreate(&multiplication_start);
    cudaEventCreate(&multiplication_end);

    cudaEventRecord(multiplication_start);

    const int block_size = 32;
    const dim3 grid_dim((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);
    const dim3 block_dim(block_size, block_size);

    // Takes just 3ms for n=1000
    multiplyMatrices<<<grid_dim, block_dim>>>(matrix_a, matrix_b, matrix_c, n, n, n);

    cudaEventRecord(multiplication_end);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, multiplication_start, multiplication_end);
    printf("GPU multiplied in %.3f milliseconds\n", milliseconds);

    if (n < 10) {
        printMatrix(matrix_c, n, n);
    }

    // Let's be nice and clean up
    cudaEventDestroy(multiplication_start);
    cudaEventDestroy(multiplication_end);

    cudaFree(matrix_a);
    cudaFree(matrix_b);
    cudaFree(matrix_c);
}

void testCPUMultiplication(int n) {
    float *matrix_a = allocateMatrix(n, n);
    float *matrix_b = allocateMatrix(n, n);
    float *matrix_c = allocateMatrix(n, n);

    setMatrixRandom(matrix_a, n, n);
    setMatrixRandom(matrix_b, n, n);

    struct timeval start, end;

    gettimeofday(&start, NULL);
    // Takes 553ms for n=1000 with -O3 optimizations
    cpuMultiplyMatrices(matrix_a, matrix_b, matrix_c, n, n, n);
    gettimeofday(&end, NULL);

    double elapsed_milliseconds =
        (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
    printf("CPU multiplied in %.3f milliseconds\n", elapsed_milliseconds);

    free(matrix_a);
    free(matrix_b);
    free(matrix_c);
}

int main() {
    int n = 1000;

    testGPUMultiplication(n);
    testCPUMultiplication(n);

    return 0;
}
