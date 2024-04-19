%: %.cu
	nvcc -O3 $< -o main
