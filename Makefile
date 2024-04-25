%: %.cu
	nvcc -O3 -lcublas $< -o main
