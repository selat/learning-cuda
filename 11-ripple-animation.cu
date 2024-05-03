#include <SDL2/SDL.h>
#include <SDL2/SDL_surface.h>
#include <SDL2/SDL_video.h>
#include <stdio.h>
#include <sys/time.h>

/*
* Render Ripple animation
*/

__global__ void kernelRippleAnimation(int tick, int width, int height, uint32_t* pixels) {
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;
    if (xid >= width || yid >= height) {
        return;
    }

    int pixel_id = yid * width + xid;

    float fx = xid - width * 0.5f;
    float fy = yid - height * 0.5f;
    float distance = hypotf(fx, fy);
    float coefficient = cosf(distance / 10.0f - tick / 100.0f) / (distance / 100.0f + 1.0f);

    uint8_t pixel_value = 128.0f + 127.0f * coefficient;
    pixels[pixel_id] = pixel_value | (pixel_value << 8) | (pixel_value << 16);
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

    SDL_Event event;
    bool is_running = true;
    int tick = 0;
    while (is_running) {
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_QUIT:
                is_running = 0;
                break;
            }
        }

        kernelRippleAnimation<<<grid_dim, block_dim>>>(tick, width, height, device_buffer);
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        }

        cudaMemcpy(window_surface->pixels, device_buffer, sizeof(uint32_t) * width * height, cudaMemcpyDeviceToHost);

        SDL_UpdateWindowSurface(window);

        ++tick;
    }
    return 0;
}
