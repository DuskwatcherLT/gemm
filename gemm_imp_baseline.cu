#include "gemm.cuh"
#include "utils.cuh"
#include <cuda.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;

using namespace nvcuda;


template<typename Type_A, typename Type_B, typename Type_C>
__global__ void gemm_baseline(const  Type_A * __restrict__ A, const Type_B *__restrict__ B, Type_C *__restrict__ C, size_t M, size_t N, size_t K) {
    // you can change everything in this function, including the function signature
    // You can create a CUDA (.cu) file containing a class that inherits from the abstract base class GEMM.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float result = 0.0f;

        for (int k = 0; k < K; ++k) {
            float a = __half2float(A[row * K + k]);
            float b = __half2float(B[k * N + col]);
            result += a * b;
        }

        C[row * N + col] = result;
    }
}


class GEMM_Baseline : public GEMM<half, half, float>  {
public:
    GEMM_Baseline(int M, int N, int K) : GEMM(M, N, K) {}

    virtual void call_kernel() override {
        dim3 num_blocks((N + 15) / 16, (M + 15) / 16);
        dim3 block_size(16,16);
        
        gemm_baseline<<<num_blocks, block_size>>>(d_A, d_B, d_C, M, N, K);        

        // configure and call your kernel here. 
        // you may also add your timer in here
        // you can also add timers elsewhere. 
    }
};

 
int main() {
    GEMM_Baseline gemm(256, 256, 256);
    bool correct = gemm.gemm();

    if (correct) {
        cout << "correct" << endl;
    } else {
        cout << "incorrect" << endl;
    }
    return 0;
}
