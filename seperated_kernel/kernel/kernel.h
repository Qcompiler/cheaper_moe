#pragma once 
 
#include <cuda_runtime.h>
#include <stddef.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cublasLt.h>


__device__ __forceinline__ float warpReduceSum(float sum,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}


void FindRowScaleBF16(  __nv_bfloat16 *x,   float *scaleRow,
                         int rows, int cols , int8_t * out, cudaStream_t stream) ;


void FindRowScaleFloat(  float *x,   float *scaleRow,
                         int rows, int cols, int8_t * out, cudaStream_t stream);


void mixgemm_sparse_fp16_dense_weight_cu(int M, int N, int K, int num_ind,
                                        half* A, half* B, half* out, 
                                        int* ind, cudaStream_t stream);

void zero_copy_cu(int *flag,  half* input);
void find_zeros_cu(int * dev_ptr, half* input, int bs, int seq, int hidden, half* last_input);

void reuse_output_cu(int * dev_ptr, half* input, int bs, int seq, int hidden, half* last_input);

void ExtractOutliersAndSetToZeros_(int M, int N, const half * A, half *fp_A, 
        const int *ind, const int len, cudaStream_t stream);
void gemmfp16_(
    const half * mat1,
    const half * mat2, half *mat3, int m, int n, int k, 
    cublasHandle_t handle, cudaStream_t stream);


void int8FusedDequantizeCUDA_(const int8_t *A,
                             const int8_t *B,
                             const float *scale_row,
                             const float *scale_col,
                             half *y, half *D, 
                             int M, int N, int K,
                             char * workspace,
                             cudaStream_t stream);
void int8quant_(int rows, int cols, const half * src, int8_t *output, 
        half *scale, cudaStream_t stream);

void int8quant_with_cached_scaling(int rows, int cols, const half * src, int8_t *output, 
        float *scale, cudaStream_t stream);
        
void int8quant_(int rows, int cols, const half * src, int8_t *output, 
        float *scale, cudaStream_t stream);

// simple gemv
void gemv_cu(int m, int n, int k,  half * vec_data_,
                                  half * mat_data_, 
                                  half * result_data_,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y , 
                                   cudaStream_t stream); 


void gemv_int4_cu(int m, int n, int k, half * vec_data_,
                                  void * mat_data_, 
                                 half * result_data_,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                    float * scaling_data_,
                                    cudaStream_t stream
                                   );  

void gemv_int4_sm90_cu(int m, int n, int k, half * vec_data_ ,
                                  void * mat_data_, 
                                   half * result_data_,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                    float * scaling_data_,
                                    cudaStream_t stream
                                   );

void gemv_int4_fp16_mix_cu(int m, int n, int k,  half * vec_data_ ,
                                  void * mat_data_, 
                                  half * result_data_,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                   float * scaling_data_,
                                    half * weight_cache_data,
                                    int * ind_data,
                                    int n_outliers,
                                    cudaStream_t stream
                                   );

void gemv_int4_fp16_mix_sm90_cu_new(int m, int n, int k,  half * vec_data_ ,
                                  void * mat_data_, 
                                  half * result_data_,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                   float * scaling_data_,
                                    half * weight_cache_data,
                                    int * ind_data,
                                    int n_outliers,
                                    cudaStream_t stream
                                   );
void gemv_int4_fp16_mix_sm90_cu(int m, int n, int k, half * vec_data_,
                                  void * mat_data_, 
                                  half * result_data_,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                   float * scaling_data_,
                                    half * weight_cache_data,
                                    int * ind_data,
                                    int n_outliers,
                                   cudaStream_t stream
                                   );


void dequant_int4_cu( int n, int k,   void * mat_data_, 
                                 half * result_data_,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                    float * scaling_data_,
                                    cudaStream_t stream, int groupsize
                                   );