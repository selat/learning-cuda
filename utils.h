#ifndef _LEARNING_CUDA_UTILS_H_
#define _LEARNING_CUDA_UTILS_H_

#define HANDLE_CUDA_ERROR(expression)                                                              \
    {                                                                                              \
        cudaError_t error = (expression);                                                          \
        if (error != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", (__FILE__), (__LINE__),                   \
                    cudaGetErrorString(error));                                                    \
            exit(1);                                                                               \
        }                                                                                          \
    }

#endif
