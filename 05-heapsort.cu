#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/*
Sort a vector using heap sort

Current timings for n=50_000_000:
qsort CPU sorted in 3284.454ms
Heap CPU sorted in 5791.861ms
GPU sorted in 1758.085ms, GPU time 716.513ms
*/

typedef struct {
    int key;
    int value;
} PriorityQueueElement;

typedef struct {
    PriorityQueueElement *elements;
    int size;
    int max_size;
} PriorityQueue;

void init(PriorityQueue *queue, int max_size) {
    queue->elements = (PriorityQueueElement *)malloc(max_size * sizeof(PriorityQueueElement));
    queue->size = 0;
    queue->max_size = max_size;
}

void push(PriorityQueue *queue, int key, int value) {
    if (queue->size == queue->max_size) {
        fprintf(stderr, "Queue is full\n");
        exit(1);
    }

    queue->elements[queue->size++] = PriorityQueueElement{key, value};
    int i = queue->size - 1;
    int next_i = (i - 1) / 2;
    while (i > 0 && queue->elements[i].key < queue->elements[next_i].key) {
        PriorityQueueElement temp = queue->elements[i];
        queue->elements[i] = queue->elements[next_i];
        queue->elements[next_i] = temp;
        i = next_i;
        next_i = (i - 1) / 2;
    }
}

PriorityQueueElement pop(PriorityQueue *queue) {
    if (queue->size == 0) {
        fprintf(stderr, "Queue is empty\n");
        exit(1);
    }

    PriorityQueueElement result = queue->elements[0];
    queue->elements[0] = queue->elements[--queue->size];
    int i = 0;
    while (1) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int smallest = i;
        if (left < queue->size && queue->elements[left].key < queue->elements[smallest].key) {
            smallest = left;
        }
        if (right < queue->size && queue->elements[right].key < queue->elements[smallest].key) {
            smallest = right;
        }
        if (smallest == i) {
            break;
        }
        PriorityQueueElement temp = queue->elements[i];
        queue->elements[i] = queue->elements[smallest];
        queue->elements[smallest] = temp;
        i = smallest;
    }
    return result;
}

int compareInts(const void *a, const void *b) {
    const int *x = (const int *)a;
    const int *y = (const int *)b;
    return *x - *y;
}

float getElapsedMilliseconds(struct timeval* start, struct timeval* end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 + (end->tv_usec - start->tv_usec) / 1000.0;
}

__global__ void kernelBubbleSort(int *vector, int n) {
    if (threadIdx.x == 0) {
        int range_start = blockDim.x * blockIdx.x;
        int range_end = blockDim.x * (blockIdx.x + 1);
        if (range_end > n) {
            range_end = n;
        }
        for (int i = range_start; i < range_end; ++i) {
            for (int j = i + 1; j < range_end; ++j) {
                if (vector[i] > vector[j]) {
                    int temp = vector[i];
                    vector[i] = vector[j];
                    vector[j] = temp;
                }
            }
        }
    }
}

// Assuming than n is divisible by 512, otherwise the last block needs to be handled separately
__global__ void kernelBitonicSort(int *vector, int n) {
    __shared__ int local_vector[512];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    local_vector[tid] = vector[idx];
    __syncthreads();

    for (unsigned int k = 2; k <= blockDim.x; k *= 2) {
        for (unsigned int j = k / 2; j > 0; j /= 2) {
            unsigned int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    if (local_vector[tid] > local_vector[ixj]) {
                        int temp = local_vector[tid];
                        local_vector[tid] = local_vector[ixj];
                        local_vector[ixj] = temp;
                    }
                } else {
                    if (local_vector[tid] < local_vector[ixj]) {
                        int temp = local_vector[tid];
                        local_vector[tid] = local_vector[ixj];
                        local_vector[ixj] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }

    vector[idx] = local_vector[tid];
}

void heapSort(int *vector, int n) {
    PriorityQueue queue;
    init(&queue, n);
    for (int i = 0; i < n; i++) {
        push(&queue, vector[i], 0);
    }
    for (int i = 0; i < n; i++) {
        vector[i] = pop(&queue).key;
    }
}

int main() {
    int n = 512 * 1000;
    int *vector = (int *)malloc(n * sizeof(int));
    int *heap_vector = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        vector[i] = rand();
        heap_vector[i] = vector[i];
    }

    int *gpu_vector;
    cudaMallocManaged(&gpu_vector, n * sizeof(int));
    for (int i = 0; i < n; i++) {
        gpu_vector[i] = vector[i];
    }

    struct timeval start, end;

    gettimeofday(&start, NULL);
    qsort(vector, n, sizeof(int), compareInts);
    gettimeofday(&end, NULL);

    printf("qsort CPU sorted in %.3fms\n", getElapsedMilliseconds(&start, &end));

    gettimeofday(&start, NULL);
    heapSort(heap_vector, n);
    gettimeofday(&end, NULL);
    printf("Heap CPU sorted in %.3fms\n", getElapsedMilliseconds(&start, &end));

    for (int i = 0; i < n; ++i) {
        if (vector[i] != heap_vector[i]) {
            printf("Error at index %d: %d != %d\n", i, vector[i], gpu_vector[i]);
            break;
        }
    }

    cudaEvent_t cuda_start, cuda_end;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_end);

    gettimeofday(&start, NULL);
    cudaEventRecord(cuda_start);
    const int threads_per_block = 512;
    const int blocks_num = (n + threads_per_block - 1) / threads_per_block;
    // kernelBubbleSort<<<blocks_num, threads_per_block>>>(gpu_vector, n);
    kernelBitonicSort<<<blocks_num, threads_per_block>>>(gpu_vector, n);
    cudaEventRecord(cuda_end);
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    }

    PriorityQueue queue;
    init(&queue, blocks_num);
    int *gpu_output_vector = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < blocks_num; ++i) {
        push(&queue, gpu_vector[i * threads_per_block], i * threads_per_block);
    }

    for (int i = 0; i < n; ++i) {
        PriorityQueueElement element = pop(&queue);

        int next_index = element.value + 1;
        if (next_index % threads_per_block != 0 && next_index < n) {
            push(&queue, gpu_vector[next_index], next_index);
        }
        gpu_output_vector[i] = element.key;
    }

    gettimeofday(&end, NULL);

    for (int i = 0; i < n; ++i) {
        if (vector[i] != gpu_output_vector[i]) {
            printf("Error at index %d: %d != %d\n", i, vector[i], gpu_vector[i]);
            break;
        }
    }

    float gpu_milliseconds = 0;
    cudaEventElapsedTime(&gpu_milliseconds, cuda_start, cuda_end);

    printf("GPU sorted in %.3fms, GPU time %.3fms\n", getElapsedMilliseconds(&start, &end), gpu_milliseconds);

    free(gpu_output_vector);
    cudaFree(gpu_vector);
    free(heap_vector);
    free(vector);
    return 0;
}
