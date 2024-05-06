#include <SDL2/SDL.h>
#include <SDL2/SDL_surface.h>
#include <SDL2/SDL_video.h>
#include <cmath>
#include <stdio.h>
#include <sys/time.h>

#include "utils.h"

/*
 * Render spheres using ray tracing
 */

struct Sphere {
    float x, y, z;
    float radius;
    float vx, vy, vz;
    float r, g, b;

    __device__ float hit(float ox, float oy, float *n) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / radius;
            return dz + z;
        }
        return -INFINITY;
    }
};

__global__ void kernelMoveSpheres(Sphere *spheres) {
    int id = blockIdx.x;

    spheres[id].x += spheres[id].vx;
    if (spheres[id].x < -1.0f) {
        spheres[id].x = -1.0f;
        spheres[id].vx = -spheres[id].vx;
    }
    if (spheres[id].x > 1.0f) {
        spheres[id].x = 1.0f;
        spheres[id].vx = -spheres[id].vx;
    }

    spheres[id].y += spheres[id].vy;
    if (spheres[id].y < -1.0f) {
        spheres[id].y = -1.0f;
        spheres[id].vy = -spheres[id].vy;
    }
    if (spheres[id].y > 1.0f) {
        spheres[id].y = 1.0f;
        spheres[id].vy = -spheres[id].vy;
    }

    spheres[id].z += spheres[id].vz;
    if (spheres[id].z < -1.0f) {
        spheres[id].z = -1.0f;
        spheres[id].vz = -spheres[id].vz;
    }
    if (spheres[id].z > 1.0f) {
        spheres[id].z = 1.0f;
        spheres[id].vz = -spheres[id].vz;
    }
}

__global__ void kernelRayTracingSpheres(int n, Sphere *spheres, int width, int height, uint32_t *pixels) {
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;
    if (xid >= width || yid >= height) {
        return;
    }

    int pixel_id = yid * width + xid;

    float fx = 2.0f * (xid - int(width / 2)) / width;
    float fy = 2.0f * (yid - int(height / 2)) / height;

    float maxz = -INFINITY;
    float light = 0.0f;
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    for (int i = 0; i < n; ++i) {
        float curz = spheres[i].hit(fx, fy, &light);
        if (curz > maxz) {
            maxz = curz;
            r = spheres[i].r * light;
            g = spheres[i].g * light;
            b = spheres[i].b * light;
        }
    }

    uint8_t pixel_r = (uint8_t)0xff * r;
    uint8_t pixel_g = (uint8_t)0xff * g;
    uint8_t pixel_b = (uint8_t)0xff * b;
    pixels[pixel_id] = pixel_b | (pixel_g << 8) | (pixel_r << 16);
}

int main(int argc, char **argv) {
    int n = 50;

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

    Sphere *spheres = (Sphere *)malloc(n * sizeof(Sphere));
    Sphere *device_spheres;
    HANDLE_CUDA_ERROR(cudaMalloc(&device_spheres, n * sizeof(Sphere)));
    for (int i = 0; i < n; ++i) {
        spheres[i].x = (rand() % 2000 - 1000) / 1000.0f;
        spheres[i].y = (rand() % 2000 - 1000) / 1000.0f;
        spheres[i].z = (rand() % 2000 - 1000) / 1000.0f;
        spheres[i].radius = (rand() % 600 + 400) / 1000.0f * 0.2f;
        spheres[i].vx = (rand() % 2000 - 1000) / 1000.0f * 0.0004f;
        spheres[i].vy = (rand() % 2000 - 1000) / 1000.0f * 0.0004f;
        spheres[i].vz = (rand() % 2000 - 1000) / 1000.0f * 0.0004f;
        spheres[i].r = (float)rand() / RAND_MAX;
        spheres[i].g = (float)rand() / RAND_MAX;
        spheres[i].b = (float)rand() / RAND_MAX;
    }
    HANDLE_CUDA_ERROR(cudaMemcpy(device_spheres, spheres, n * sizeof(Sphere), cudaMemcpyHostToDevice));

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
        kernelMoveSpheres<<<n, 1>>>(device_spheres);
        kernelRayTracingSpheres<<<grid_dim, block_dim>>>(n, device_spheres, width, height, device_buffer);
        HANDLE_CUDA_ERROR(cudaMemcpy(window_surface->pixels, device_buffer,
                                     sizeof(uint32_t) * width * height, cudaMemcpyDeviceToHost));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Rendered in %.3f ms\n", milliseconds);

        SDL_UpdateWindowSurface(window);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(device_buffer);
    cudaFree(device_spheres);

    free(spheres);
    return 0;
}
