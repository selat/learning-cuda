#include <stdio.h>

/*
Print information about CUDA devices

Example output:
Number of CUDA devices: 1

Device 0: NVIDIA GeForce RTX 3090
  Compute capability: 8.6
  Number of multiprocessors: 82
  Maximum threads per block: 1024
  Concurrent kernels: 1
  Maximum dimensions of block: [1024, 1024, 64]
  Maximum dimensions of grid: [2147483647, 65535, 65535]

  Total global memory: 25430786048 bytes
  Total constant memory: 65536 bytes
  Shared memory per block: 49152 bytes
  Memory bus width: 384 bits
  Registers per block: 65536
  L2 cache size: 6291456 bytes
  Warp size: 32
  Clock rate: 1695000 kHz
  Memory clock rate: 9751000 kHz
  Can map host memory: Yes
  ECC enabled: No
*/

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    printf("Number of CUDA devices: %d\n\n", device_count);

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp device_properties;
        cudaGetDeviceProperties(&device_properties, i);
        printf("Device %d: %s\n", i, device_properties.name);

        printf("  Compute capability: %d.%d\n", device_properties.major, device_properties.minor);
        printf("  Number of multiprocessors: %d\n", device_properties.multiProcessorCount);
        printf("  Maximum threads per block: %d\n", device_properties.maxThreadsPerBlock);
        printf("  Concurrent kernels: %d\n", device_properties.concurrentKernels);
        printf("  Maximum dimensions of block: [%d, %d, %d]\n", device_properties.maxThreadsDim[0],
               device_properties.maxThreadsDim[1], device_properties.maxThreadsDim[2]);
        printf("  Maximum dimensions of grid: [%d, %d, %d]\n\n", device_properties.maxGridSize[0],
               device_properties.maxGridSize[1], device_properties.maxGridSize[2]);

        printf("  Total global memory: %lu bytes\n", device_properties.totalGlobalMem);
        printf("  Total constant memory: %lu bytes\n", device_properties.totalConstMem);
        printf("  Shared memory per block: %lu bytes\n", device_properties.sharedMemPerBlock);
        printf("  Memory bus width: %d bits\n", device_properties.memoryBusWidth);
        printf("  Registers per block: %d\n", device_properties.regsPerBlock);
        printf("  L2 cache size: %d bytes\n", device_properties.l2CacheSize);
        printf("  Warp size: %d\n", device_properties.warpSize);
        printf("  Clock rate: %d kHz\n", device_properties.clockRate);
        printf("  Memory clock rate: %d kHz\n", device_properties.memoryClockRate);
        printf("  Can map host memory: %s\n", device_properties.canMapHostMemory ? "Yes" : "No");
        printf("  ECC enabled: %s\n", device_properties.ECCEnabled ? "Yes" : "No");
    }

    return 0;
}
