09-n-body-problem: 09-n-body-problem.cu
	nvcc 09-n-body-problem.cu -lSDL2 -lGLEW -lGL -o main

10-julia-set: 10-julia-set.cu
	nvcc 10-julia-set.cu -lSDL2 -o main

11-ripple-animation: 11-ripple-animation.cu
	nvcc 11-ripple-animation.cu -lSDL2 -o main

12-ray-tracing-spheres: 12-ray-tracing-spheres.cu
	nvcc 12-ray-tracing-spheres.cu -lSDL2 -o main

13-heat-transfer: 13-heat-transfer.cu
	nvcc 13-heat-transfer.cu -lSDL2 -o main

%: %.cu
	nvcc -O3 -lcublas $< -o main
