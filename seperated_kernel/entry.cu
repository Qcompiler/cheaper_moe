#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <cuda_fp16.h>

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>

#include "kernel.h"



void gemv(int m, int n, int k, const torch::Tensor  &x,
                                  const torch::Tensor &weight, 
                                  torch::Tensor _out,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y ){
                      

  half * vec_data_ = reinterpret_cast<half*>(x.data_ptr<at::Half>());                                  
  half * mat_data_ = reinterpret_cast<half*>(weight.data_ptr<at::Half>());
  
  half * result_data_ = reinterpret_cast<half*>(_out.data_ptr<at::Half>());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  gemv_cu(m, n, k, vec_data_, mat_data_, result_data_, block_dim_x, block_dim_y, stream);
  return  ;
}



void gemv_int4(int m, int n, int k, const torch::Tensor  &x,
                                  const torch::Tensor &weight, 
                                  torch::Tensor _out,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                    torch::Tensor scaling
                                   ){
 

  void * mat_data_ = (weight.data_ptr());
  half * vec_data_ = reinterpret_cast<half*>(x.data_ptr<at::Half>());
  half * result_data_ = reinterpret_cast<half*>(_out.data_ptr<at::Half>());
  float * scaling_data_ = reinterpret_cast<float*>(scaling.data_ptr<float>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
 
  gemv_int4_cu(m, n, k, vec_data_, mat_data_, result_data_, block_dim_x, block_dim_y, scaling_data_, stream);
   
  return  ;
}









void gemv_int4_fp16_mix(int m, int n, int k, const torch::Tensor  &x,
                                  const torch::Tensor &weight, 
                                  torch::Tensor _out,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                    torch::Tensor scaling,
                                    torch::Tensor weight_cache,
                                    torch::Tensor ind,
                                    int n_outliers
                                   ){                      

  void * mat_data_ = (weight.data_ptr());
  half * vec_data_ = reinterpret_cast<half*>(x.data_ptr<at::Half>());
  half * result_data_ = reinterpret_cast<half*>(_out.data_ptr<at::Half>());
  float * scaling_data_ = reinterpret_cast<float*>(scaling.data_ptr<float>());

  half * weight_cache_data = reinterpret_cast<half*>(weight_cache.data_ptr<at::Half>());
  int * ind_data = reinterpret_cast<int*>(ind.data_ptr<int>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  gemv_int4_fp16_mix_cu( m,  n,  k, 
                          vec_data_,
                          mat_data_, 
                          result_data_,
                          block_dim_x,
                          block_dim_y,
                          scaling_data_,
                          weight_cache_data,
                          ind_data, n_outliers, stream);

 
  return  ;
}


void gemv_int4_sm90(int m, int n, int k, const torch::Tensor  &x,
                                  const torch::Tensor &weight, 
                                  torch::Tensor _out,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                    torch::Tensor scaling
                                   ){
 
  void * mat_data_ = (weight.data_ptr());
  half * vec_data_ = reinterpret_cast<half*>(x.data_ptr<at::Half>());
  half * result_data_ = reinterpret_cast<half*>(_out.data_ptr<at::Half>());
  float * scaling_data_ = reinterpret_cast<float*>(scaling.data_ptr<float>());

 cudaStream_t stream = at::cuda::getCurrentCUDAStream();
 gemv_int4_sm90_cu( m,  n,  k,  vec_data_ ,
                  mat_data_, 
                  result_data_,
                  block_dim_x,
                  block_dim_y,
                  scaling_data_,
                  stream );
  return  ;
}


void gemv_int4_fp16_mix_sm90(int m, int n, int k, const torch::Tensor  &x,
                                  const torch::Tensor &weight, 
                                  torch::Tensor _out,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                    torch::Tensor scaling,
                                    torch::Tensor weight_cache,
                                    torch::Tensor ind,
                                    int n_outliers
                                   ){                      

  void * mat_data_ = (weight.data_ptr());
  half * vec_data_ = reinterpret_cast<half*>(x.data_ptr<at::Half>());
  half * result_data_ = reinterpret_cast<half*>(_out.data_ptr<at::Half>());
  float * scaling_data_ = reinterpret_cast<float*>(scaling.data_ptr<float>());

  half * weight_cache_data = reinterpret_cast<half*>(weight_cache.data_ptr<at::Half>());
  int * ind_data = reinterpret_cast<int*>(ind.data_ptr<int>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  gemv_int4_fp16_mix_sm90_cu( m,  n,  k, 
                          vec_data_,
                          mat_data_, 
                          result_data_,
                          block_dim_x,
                          block_dim_y,
                          scaling_data_,
                          weight_cache_data,
                          ind_data, n_outliers, stream);

 
  return  ;
}


void gemv_int4_fp16_mix_sm90_new(int m, int n, int k, const torch::Tensor  &x,
                                  const torch::Tensor &weight, 
                                  torch::Tensor _out,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                    torch::Tensor scaling,
                                    torch::Tensor weight_cache,
                                    torch::Tensor ind,
                                    int n_outliers
                                   ){                      

  void * mat_data_ = (weight.data_ptr());
  half * vec_data_ = reinterpret_cast<half*>(x.data_ptr<at::Half>());
  half * result_data_ = reinterpret_cast<half*>(_out.data_ptr<at::Half>());
  float * scaling_data_ = reinterpret_cast<float*>(scaling.data_ptr<float>());

  half * weight_cache_data = reinterpret_cast<half*>(weight_cache.data_ptr<at::Half>());
  int * ind_data = reinterpret_cast<int*>(ind.data_ptr<int>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  gemv_int4_fp16_mix_sm90_cu_new( m,  n,  k, 
                          vec_data_,
                          mat_data_, 
                          result_data_,
                          block_dim_x,
                          block_dim_y,
                          scaling_data_,
                          weight_cache_data,
                          ind_data, n_outliers, stream);

 
  return  ;
}





torch::Tensor mixgemmforward_direct(int M, int N, int K, 
                            torch::Tensor & A_, 
                            torch::Tensor & scale_A_,
                            torch::Tensor &w_, torch::Tensor &s_,
                            int batch, int seq_len ){

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(A_.device());  
    auto fp16_out = torch::zeros(
        {batch, seq_len,  N }, options);

    half * Out = reinterpret_cast<half *>(fp16_out.data_ptr<at::Half>());

    auto options_i8 = torch::TensorOptions().dtype(torch::kInt8).device(A_.device());
    auto quant_out = torch::zeros(
      {M, K }, options_i8);
    int8_t* int8_out = reinterpret_cast<int8_t *>(quant_out.data_ptr<int8_t>());


    // auto options_float = torch::TensorOptions().dtype(torch::kFloat).device(A_.device()); 
    // auto scale_a_tmp = torch::zeros(
    //     {M, 1 }, options_float);
    float* scale_a = reinterpret_cast<float *>(scale_A_.data_ptr<float>());

 

    const half * A = reinterpret_cast<half const*>(A_.data_ptr<at::Half>());

    const int8_t * W = reinterpret_cast<int8_t const*>(w_.data_ptr<int8_t>());

    const float * scale_b = reinterpret_cast<float const* >(s_.data_ptr<at::Half>());

 

 
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    int8quant_(M, K, A, int8_out, scale_a, stream);
    int8FusedDequantizeCUDA_(int8_out, W, scale_a,
                            scale_b, Out, Out, M, N, K, 
                            reinterpret_cast<char*>(int8_out),
                            stream);
    return fp16_out;
}


void int8quant(int M, int K, torch::Tensor & A_,  torch::Tensor &s_,
               torch::Tensor & quant_out,
               torch::Tensor & fp_activation_tmp,
               torch::Tensor & ind_,
               int num_ind){

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                


    int8_t* int8_out = reinterpret_cast<int8_t *>(quant_out.data_ptr<int8_t>());
    const half * A = reinterpret_cast<half const*>(A_.data_ptr<at::Half>());
    const int * ind  = reinterpret_cast<int const*>(ind_.data_ptr<int32_t>());   


    if (num_ind > 0) {
      
      half* fp_activation = reinterpret_cast<half *>(fp_activation_tmp.data_ptr<at::Half>());

      ExtractOutliersAndSetToZeros_(M, K, A, fp_activation, ind, num_ind, stream);
    }
    float* scale_a = reinterpret_cast< float *>(s_.data_ptr<float>());

    int8quant_(M, K, A, int8_out, scale_a, stream);

    return  ;
}


torch::Tensor mixgemmforward_dynamic(int M, int N, int K, 
                            torch::Tensor & A_, torch::Tensor &w_, torch::Tensor &s_,
                            int batch, int seq_len,
                            torch::Tensor &fp_w_, torch::Tensor & ind_, int num_ind){

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(A_.device());  
    auto fp16_out = torch::zeros(
        {batch, seq_len, N }, options);

    half * Out = reinterpret_cast<half *>(fp16_out.data_ptr<at::Half>());

    auto options_i8 = torch::TensorOptions().dtype(torch::kInt8).device(A_.device());
    auto quant_out = torch::zeros(
      {M, K }, options_i8);
    int8_t* int8_out = reinterpret_cast<int8_t *>(quant_out.data_ptr<int8_t>());

    auto options_float = torch::TensorOptions().dtype(torch::kFloat).device(A_.device()); 
    auto scale_a_tmp = torch::zeros(
        {M, 1 }, options_float);
    float* scale_a = reinterpret_cast< float *>(scale_a_tmp.data_ptr<float>());
    const float * scale_b = reinterpret_cast<const float* >(s_.data_ptr<at::Half>());


    auto fp_activation_tmp = torch::zeros(
        {M, num_ind }, options);
    half* fp_activation = reinterpret_cast<half *>(fp_activation_tmp.data_ptr<at::Half>());



    const half * A = reinterpret_cast<half const*>(A_.data_ptr<at::Half>());

    const int8_t * W = reinterpret_cast<int8_t const*>(w_.data_ptr<int8_t>());

    

    
    // outliers
    const half * fp_weight = reinterpret_cast<half const*>(fp_w_.data_ptr<at::Half>());

    const int * ind  = reinterpret_cast<int const*>(ind_.data_ptr<int32_t>());   
   


    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle,stream);
    ExtractOutliersAndSetToZeros_(M, K, A, fp_activation, ind, num_ind, stream);
    
    gemmfp16_(fp_activation, fp_weight,Out, M, N, num_ind, handle, stream);
    int8quant_(M, K, A, int8_out, scale_a, stream);


    int8FusedDequantizeCUDA_(int8_out, W, scale_a,
                            scale_b, Out, Out, M, N, K, 
                            reinterpret_cast<char*>(int8_out),
                            stream);
    return fp16_out;
}


torch::Tensor mixgemm_sparse_fp16_dense_weight(int M, int N, int K,  int num_ind,
                            torch::Tensor & A_, torch::Tensor &B_,  
                            torch::Tensor & ind_){

    half * A = reinterpret_cast<half *>(A_.data_ptr<at::Half>());
    half * B = reinterpret_cast<half *>(B_.data_ptr<at::Half>());
    int * ind  = reinterpret_cast<int *>(ind_.data_ptr<int32_t>());   

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(A_.device());  
    auto fp16_out = torch::zeros(
        {M, N }, options);

    half * Out = reinterpret_cast<half *>(fp16_out.data_ptr<at::Half>());


    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                

    mixgemm_sparse_fp16_dense_weight_cu(M, N, K, num_ind, A, B, Out, ind, stream);
    return fp16_out;

}

void zero_copy(torch::Tensor & cpu_A,   torch::Tensor & input_A){

    int * tensor = reinterpret_cast<int *>(cpu_A.data_ptr<int32_t>());

    int * dev_ptr ;


    half * input = reinterpret_cast<half *>(input_A.data_ptr<at::Half>());

    cudaHostGetDevicePointer((void **)&dev_ptr, (void *)tensor, 0);

    zero_copy_cu(dev_ptr, input);

}

void find_zeros(torch::Tensor & cpu_A,   torch::Tensor & input_A, int bs, int seq, int hidden, torch::Tensor & last_input_){
    int * tensor = reinterpret_cast<int *>(cpu_A.data_ptr<int32_t>());
    int * dev_ptr ;
    cudaHostGetDevicePointer((void **)&dev_ptr, (void *)tensor, 0);

    half * input = reinterpret_cast<half *>(input_A.data_ptr<at::Half>());
    half * last_input = reinterpret_cast<half *>(last_input_.data_ptr<at::Half>());

    find_zeros_cu(dev_ptr, input, bs, seq, hidden, last_input);

}



void reuse_output(torch::Tensor & cpu_A,   torch::Tensor & input_A, int bs, int seq, int hidden, torch::Tensor & last_input_){
    int * tensor = reinterpret_cast<int *>(cpu_A.data_ptr<int32_t>());
    int * dev_ptr ;
    cudaHostGetDevicePointer((void **)&dev_ptr, (void *)tensor, 0);

    half * input = reinterpret_cast<half *>(input_A.data_ptr<at::Half>());
    half * last_input = reinterpret_cast<half *>(last_input_.data_ptr<at::Half>());

    reuse_output_cu(dev_ptr, input, bs, seq, hidden, last_input);

}
torch::Tensor mixgemmforward_direct_with_scaling(int M, int N, int K, 
                            torch::Tensor & A_, 
                            torch::Tensor & A_scaling, 
                            torch::Tensor &w_, 
                            torch::Tensor &s_,
                            int batch, int seq_len ){

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(A_.device());  
    auto fp16_out = torch::zeros(
        {batch, seq_len,  N }, options);

    half * Out = reinterpret_cast<half *>(fp16_out.data_ptr<at::Half>());

    auto options_i8 = torch::TensorOptions().dtype(torch::kInt8).device(A_.device());
    auto quant_out = torch::zeros(
      {M, K }, options_i8);
    int8_t* int8_out = reinterpret_cast<int8_t *>(quant_out.data_ptr<int8_t>());



    float* scale_a = reinterpret_cast<float *>(A_scaling.data_ptr<float>());

 

    const half * A = reinterpret_cast<half const*>(A_.data_ptr<at::Half>());

    const int8_t * W = reinterpret_cast<int8_t const*>(w_.data_ptr<int8_t>());

    const float * scale_b = reinterpret_cast<float const* >(s_.data_ptr<at::Half>());

 

 
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    int8quant_with_cached_scaling(M, K, A, int8_out, scale_a, stream);
    int8FusedDequantizeCUDA_(int8_out, W, scale_a,
                            scale_b, Out, Out, M, N, K, 
                            reinterpret_cast<char*>(int8_out),
                            stream);
    return fp16_out;
}


// void dequant_int4_cu( int n, int k,   void * mat_data_, 
//                                  half * result_data_,
//                                   unsigned int block_dim_x,
//                                    unsigned int block_dim_y,
//                                     float * scaling_data_,
//                                     cudaStream_t stream
//                                    );
torch::Tensor dequant(  const torch::Tensor & q_weight,  torch::Tensor &scales,
                         int rows, int cols, int bit, int groupsize,
                          unsigned int block_dim_x,
                                   unsigned int block_dim_y){

  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(q_weight.device());  
  auto fp16_out = torch::zeros(
        {rows, cols}, options);
  
  if (groupsize == -1) {
     groupsize = cols;
  }
  dequant_int4_cu( rows, cols, (void *)q_weight.data_ptr(),
                                  reinterpret_cast<half *>(fp16_out.data_ptr<at::Half>()),
                                   block_dim_x, block_dim_y,
                                   reinterpret_cast<float *>(scales.data_ptr<float>()),
                                   at::cuda::getCurrentCUDAStream(), groupsize );
  return fp16_out;
                  
}
// void FindRowScaleFloat(  float *x,   float *scaleRow,
//                          int rows, int cols, int8_t * out, cudaStream_t stream);
torch::Tensor FindRowScaleF32(  const torch::Tensor &x,  torch::Tensor &scaleRow,
                         int rows, int cols, int bit ) {


    auto options_i8 = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    auto quant_out = torch::zeros(
      {rows, cols }, options_i8);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();      
 
    FindRowScaleFloat( (float *)x.data_ptr<float>(), (float *)scaleRow.data_ptr<float>(), 
              rows, cols, (int8_t *)quant_out.data_ptr<int8_t>(), stream);
    return quant_out;

}