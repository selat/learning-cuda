#include <SDL2/SDL.h>
#include <SDL2/SDL_surface.h>
#include <SDL2/SDL_video.h>
#include <stdio.h>
#include <sys/time.h>

#include "utils.h"

#define MAX_TEMPERATURE 1000.0f

/*
 * Simulate heat transfer
 */

__global__ void kernelUpdateHeatSources(int width, int height, float *heat_sources,
                                        float *temperatures) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int offset = y * width + x;

    if (heat_sources[offset] > 0.0f) {
        temperatures[offset] = heat_sources[offset];
    }
}

__global__ void kernelHeatTransfer(int width, int height, float heat_transfer_coefficient,
                                   float *temperature_in, float *temperature_out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int offset = y * width + x;

    float deltas_sum = 0.0f;
    if (x > 0) {
        deltas_sum += temperature_in[offset - 1];
    } else {
        deltas_sum += temperature_in[offset];
    }

    if (x < width - 1) {
        deltas_sum += temperature_in[offset + 1];
    } else {
        deltas_sum += temperature_in[offset];
    }

    if (y > 0) {
        deltas_sum += temperature_in[offset - width];
    } else {
        deltas_sum += temperature_in[offset];
    }

    if (y < height - 1) {
        deltas_sum += temperature_in[offset + width];
    } else {
        deltas_sum += temperature_in[offset];
    }
    deltas_sum -= 4.0f * temperature_in[offset];

    float new_temperature = temperature_in[offset] + heat_transfer_coefficient * deltas_sum;
    if (new_temperature > MAX_TEMPERATURE) {
        new_temperature = MAX_TEMPERATURE;
    }
    if (new_temperature < 0.0f) {
        new_temperature = 0.0f;
    }
    temperature_out[offset] = new_temperature;
}

__global__ void kernelFillTemperaturePixels(int width, int height, float *temperature,
                                            uint32_t *pixels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int offset = y * width + x;
    float t = temperature[offset];
    uint8_t r = (uint8_t)(255.0f * t / MAX_TEMPERATURE);
    uint8_t g = 0;
    uint8_t b = 0;
    pixels[offset] = (r << 16) | (g << 8) | b;
}

int main(int argc, char **argv) {
    const float heat_transfer_coefficient = 0.5f;
    const int iteration_per_frame = 10;

    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *window = SDL_CreateWindow("Ray Tracing Spheres", SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED, 1024, 1024, 0);
    if (window == NULL) {
        fprintf(stderr, "Failed to create window: %s", SDL_GetError());
        return 1;
    }

    int width, height;
    SDL_GetWindowSize(window, &width, &height);
    printf("%d x %d\n", width, height);

    SDL_Surface *window_surface = SDL_GetWindowSurface(window);

    uint32_t *device_buffer;
    HANDLE_CUDA_ERROR(cudaMalloc(&device_buffer, sizeof(uint32_t) * width * height));

    float *temperatures = (float *)malloc(sizeof(float) * width * height);
    for (int i = 0; i < width * height; ++i) {
        temperatures[i] = 0.0f;
    }
    for (int i = 0; i < width; ++i) {
        temperatures[i] = 1000.0f;
    }

    float *device_heat_sources;
    HANDLE_CUDA_ERROR(cudaMalloc(&device_heat_sources, sizeof(float) * width * height));
    HANDLE_CUDA_ERROR(cudaMemcpy(device_heat_sources, temperatures, sizeof(float) * width * height,
                                 cudaMemcpyHostToDevice));

    float *device_temperatures_in;
    HANDLE_CUDA_ERROR(cudaMalloc(&device_temperatures_in, sizeof(float) * width * height));
    HANDLE_CUDA_ERROR(cudaMemcpy(device_temperatures_in, temperatures,
                                 sizeof(float) * width * height, cudaMemcpyHostToDevice));

    float *device_temperatures_out;
    HANDLE_CUDA_ERROR(cudaMalloc(&device_temperatures_out, sizeof(float) * width * height));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int block_size = 16;
    dim3 grid_dim(width / block_size, height / block_size, 1);
    dim3 block_dim(block_size, block_size, 1);

    SDL_Event event;
    bool is_running = true;
    while (is_running) {
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_QUIT:
                is_running = 0;
                break;
            }
        }

        cudaEventRecord(start);

        for (int i = 0; i < iteration_per_frame; ++i) {
            kernelUpdateHeatSources<<<grid_dim, block_dim>>>(width, height, device_heat_sources,
                                                            device_temperatures_in);
            kernelHeatTransfer<<<grid_dim, block_dim>>>(width, height, heat_transfer_coefficient,
                                                        device_temperatures_in,
                                                        device_temperatures_out);
            kernelFillTemperaturePixels<<<grid_dim, block_dim>>>(width, height, device_temperatures_out,
                                                                device_buffer);
            HANDLE_CUDA_ERROR(cudaMemcpy(window_surface->pixels, device_buffer,
                                        sizeof(uint32_t) * width * height, cudaMemcpyDeviceToHost));

            float *tmp = device_temperatures_in;
            device_temperatures_in = device_temperatures_out;
            device_temperatures_out = tmp;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Simulated in %.3f ms\n", milliseconds);

        SDL_UpdateWindowSurface(window);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(device_buffer);
    cudaFree(device_temperatures_in);
    cudaFree(device_temperatures_out);

    free(temperatures);
    return 0;
}
