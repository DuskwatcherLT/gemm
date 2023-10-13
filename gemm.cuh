#ifndef HOMEWORK_GEMM_CUH
#define HOMEWORK_GEMM_CUH

#include "utils.cuh"
#include <cstring>
#include <iostream>

using std::cout;
using std::endl;

/**
 * Prepare data and 
 * Compute C = AB
 * and validate the results
 * 
 * C is M x N
 * A is M x K
 * B is K x N
 * 
*/
template <typename A_Type, typename B_Type, typename C_Type>
class GEMM {
public: 
    GEMM(int M, int N, int K) : M(M), N(N), K(K), ref_result(nullptr) {
        h_A = new A_Type[M * K];
        h_B = new B_Type[K * N];
        h_C = new C_Type[M * N];
        memset(h_A, sizeof(A_Type) * M * K, 0);
        memset(h_B, sizeof(B_Type) * K * N, 0);
        memset(h_C, sizeof(A_Type) * M * N, 0);                

        CHECK_ERROR(cudaMalloc(&d_A, sizeof(A_Type) * M * K));
        CHECK_ERROR(cudaMalloc(&d_B, sizeof(B_Type) * K * N));
        CHECK_ERROR(cudaMalloc(&d_C, sizeof(C_Type) * M * N));
        
        CHECK_ERROR(cudaMemcpy(d_A, h_A, sizeof(A_Type) * M * K, cudaMemcpyHostToDevice)); 
        CHECK_ERROR(cudaMemcpy(d_B, h_B, sizeof(B_Type) * K * N, cudaMemcpyHostToDevice));
        CHECK_ERROR(cudaMemcpy(d_C, h_C, sizeof(C_Type) * M * N, cudaMemcpyHostToDevice));
    }

    virtual ~GEMM() {
        delete [] h_A;
        delete [] h_B;
        delete [] h_C;
        CHECK_ERROR(cudaFree(d_A));
        CHECK_ERROR(cudaFree(d_B));
        CHECK_ERROR(cudaFree(d_C));

        if (ref_result) {
            delete[] ref_result;
        }
    }

    virtual __host__ bool gemm() {
        prepare_data();
        copy_data_to_device();
        call_kernel();
        copy_data_back();
        compute_reference_result();
        return validate();
    }

    
protected:
    virtual void call_kernel() = 0; 


    virtual void __host__ prepare_data() {
        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(0, 2);

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                h_A[i * K + j] = static_cast<A_Type>(dist(e2));
            }
        }

        for (int i = 0; i < K; i++) {
            for (int j = 0; j < N; j++) {
                h_B[i * N + j] = static_cast<B_Type>(dist(e2));
            }
        }
    }

    void __host__ copy_data_to_device() {
        CHECK_ERROR(cudaMemcpy(d_A, h_A, sizeof(A_Type) * M * K, cudaMemcpyHostToDevice)); 
        CHECK_ERROR(cudaMemcpy(d_B, h_B, sizeof(B_Type) * K * N, cudaMemcpyHostToDevice));
        CHECK_ERROR(cudaMemcpy(d_C, h_C, sizeof(C_Type) * M * N, cudaMemcpyHostToDevice));
    }

    virtual __host__ void copy_data_back() {
        cudaDeviceSynchronize();
        GET_LAST_ERR();
        CHECK_ERROR(cudaMemcpy(h_C, d_C, sizeof(C_Type) * M * N, cudaMemcpyDeviceToHost));
    }



    virtual __host__ void compute_reference_result() {
        if (ref_result == nullptr) {
            ref_result = new float[M * N];
        }
        
        
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float ss = 0;
                for (int k = 0; k < K; k++) {
                    ss += static_cast<float> (h_A[i * N + k]) * static_cast<float>(h_B[k * N + j]); 
                }
                
                ref_result[i * N + j] = ss;
            }
        }
    }

    
    virtual __host__ bool validate() const {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++ ) {
                if (abs(ref_result[i * N + j] - h_C[i * N + j]) > 1E-2) {
                    cout << i << " " << j << " " << ref_result[i * N + j] << " " << h_C[i * N + j] << " " << endl;
                    return false;
                }
            }
        }

        return true;
    }


    const int M, N, K; 
    
    // host matrices 
    A_Type *h_A;
    B_Type *h_B;
    C_Type *h_C;

    // device matrices
    A_Type *d_A;
    B_Type *d_B;
    C_Type *d_C;

    C_Type *ref_result;
}; 



#endif