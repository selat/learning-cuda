#include <stdio.h>

/*
Sum two float arrays
*/
__global__ void add(float *x, float *y, float *output) {
  int index = blockIdx.x;
  output[index] = x[index] + y[index];
}

void printArray(float *array, int size) {
  for (int i = 0; i < size; ++i) {
    printf("%*.1f ", 4, array[i]);
  }
  puts("");
}

int main() {
  int n = 10;
  float *input_x, *input_y, *output;
  cudaMallocManaged(&input_x, n * sizeof(float));
  cudaMallocManaged(&input_y, n * sizeof(float));
  cudaMallocManaged(&output, n * sizeof(float));

  for (int i = 0; i < n; ++i) {
    input_x[i] = i;
    input_y[i] = i * i;
  }

  add<<<n, 1>>>(input_x, input_y, output);

  cudaDeviceSynchronize();

  printArray(input_x, n);
  puts("+");
  printArray(input_y, n);
  puts("=");
  printArray(output, n);

  return 0;
}
