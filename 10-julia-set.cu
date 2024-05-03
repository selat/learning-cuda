#include <SDL2/SDL.h>
#include <SDL2/SDL_surface.h>
#include <SDL2/SDL_video.h>
#include <stdio.h>
#include <sys/time.h>

/*
* Render Julia Set, which is defined as all points for which the following coverged:
* a_{n+1} = a_n * a_n + c
*/

struct cuComplex {
    float r;
    float i;

    __device__ cuComplex(float a, float b) : r(a), i(b) {}

    __device__ float magnitude2() {
        return r * r + i * i;
    }

    __device__ cuComplex operator*(const cuComplex& other) {
        return cuComplex(r * other.r - i * other.i, i * other.r + r * other.i);
    }

    __device__ cuComplex operator+(const cuComplex& other) {
        return cuComplex(r + other.r, i + other.i);
    }
};

__device__ bool juliaValue(float x, float y, float scale) {
    cuComplex c(-0.8, 0.156);
    cuComplex a(x, y);

    int iterations_num = 200 + 80 * logf(1 / scale);
    for (int i = 0; i < iterations_num; ++i) {
        a = a * a + c;
        if (a.magnitude2() > 10.0f) {
            return false;
        }
    }
    return true;
}

__global__ void kernelJuliaSet(float scale, int width, int height, uint32_t *pixels) {
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;
    if (xid >= width || yid >= height) {
        return;
    }

    int pixel_id = yid * width + xid;

    float scaled_x = scale * (2.0f * float(xid) / width - 1.0f);
    float scaled_y = scale * (2.0f * float(yid) / height - 1.0f);
    if (juliaValue(scaled_x, scaled_y, scale)) {
        pixels[pixel_id] = 0xffffff;
    } else {
        pixels[pixel_id] = 0;
    }
}

int main(int argc, char** argv) {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *window = SDL_CreateWindow("Julia Set", SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED, 1024, 1024, 0);
    if (window == NULL) {
        fprintf(stderr, "Failed to create window: %s", SDL_GetError());
        return 1;
    }

    int width, height;
    SDL_GetWindowSize(window, &width, &height);
    printf("%d x %d\n", width, height);

    SDL_Surface* window_surface = SDL_GetWindowSurface(window);

    uint32_t *device_buffer;
    cudaMalloc(&device_buffer, sizeof(uint32_t) * width * height);

    const int block_size = 16;
    dim3 grid_dim(width / block_size, height / block_size, 1);
    dim3 block_dim(block_size, block_size, 1);

    cudaEvent_t kernel_start, kernel_end;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);

    SDL_Event event;
    bool is_running = true;
    float scale = 2.0f;
    while (is_running) {
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_QUIT:
                is_running = 0;
                break;
            }
        }

        cudaEventRecord(kernel_start);
        kernelJuliaSet<<<grid_dim, block_dim>>>(scale, width, height, device_buffer);
        cudaEventRecord(kernel_end);
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        }

        float milliseconds;
        cudaEventElapsedTime(&milliseconds, kernel_start, kernel_end);
        printf("Scale %f rendered in %.3f milliseconds\n", scale, milliseconds);
        cudaMemcpy(window_surface->pixels, device_buffer, sizeof(uint32_t) * width * height, cudaMemcpyDeviceToHost);

        SDL_UpdateWindowSurface(window);

        scale *= 0.9995f;
    }

    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_end);
    cudaFree(device_buffer);
    return 0;
}
