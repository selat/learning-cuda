09-n-body-problem: 09-n-body-problem.cu
	nvcc 09-n-body-problem.cu -lSDL2 -lGLEW -lGL -o main

%: %.cu
	nvcc -O3 -lcublas $< -o main
