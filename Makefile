all: bin/gemm_baseline

bin/gemm_baseline: utils.cuh gemm.cuh gemm_imp_baseline.cu 
	mkdir -p bin
	nvcc -arch=native -G -O3 gemm_imp_baseline.cu -o bin/gemm_baseline 


# add your versions as targets in here. 
# you can also try different compiler options here. 

clean:
	rm -rf bin

.PHONY : clean

