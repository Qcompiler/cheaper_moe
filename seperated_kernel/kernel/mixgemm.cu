#include <iostream>
#include <stdexcept>
#include<cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include "symmetric/gemm/device/gemm_dequant.h"
#include "common/common.h"
#include "kernel.h"
void int8FusedDequantizeCUDA_(const int8_t *A,
                             const int8_t *B,
                             const float *scale_row,
                             const float *scale_col,
                             half *y, half *D, 
                             int M, int N, int K,
                             char * workspace,
                             cudaStream_t stream) {

 
  using Gemm = cutlass::gemm::device::symmetric::GemmDequant<
      float, //  Element for iterator 
      int8_t,                          // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      int8_t,                          // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      cutlass::half_t,                 // ElementOutput
      cutlass::layout::RowMajor,       // LayoutOutput
      int32_t,                         // ElementAccumulator
      cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
      cutlass::arch::Sm80,  // tag indicating target GPU compute architecture
      cutlass::gemm::GemmShape<128, 256, 64>
      >;

  Gemm gemmOp;
  //cutlass::Status status = gemmOp(stream);


  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {(const int8_t *)A, K},
      {(const int8_t *)B, K},
      {(const cutlass::half_t *)y, N},
      {(cutlass::half_t *)D, N},
      {(const float *)scale_col, N},
      {(const float *)scale_row, M},
      Gemm::ElementC(1)};

  gemmOp.initialize(arguments, workspace, stream);
  //status = gemmOp(arguments);
  gemmOp.run(stream);
 
}






template<int size, typename T>
__global__ void FindRowScaleKernel_(int8_t * output, const half * d_in, T * scale, int rows, int cols){

    __shared__ half sdata[size];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x ;
    if (bid > rows)
        return ;
    const  __half *start = d_in + bid * cols;
    int8_t * d_out = output + bid * cols;
    sdata[tid] = __habs(start[tid]); 
    for (int i = tid + size; i < cols; i += size)
        sdata[tid] = __hmax ( __habs(start[i]),  sdata[tid] ); 
    __syncthreads();


    // do reduction in shared mem
    for (unsigned int s= blockDim.x/2; s >= 1; s >>=1 ) {
        if (tid < s) {
            sdata[tid] =  __hmax ( __habs(sdata[tid + s]),  sdata[tid]);
        }
        __syncthreads();
    }

    half  max = sdata[0];
    // write result for this block to global mem
    //if (tid < 32) warpReduce(sdata, tid);

    __syncthreads();

    half quant_scales = __hdiv( max, 127.0);
    if (tid == 0){
        scale[bid] = (T)quant_scales;
    }
    // quant
    for (int i = tid ; i < cols; i += size)
        d_out[i] =  static_cast<int8_t>(__half2int_rn( __hdiv( start[i], quant_scales ) ))  ; 
    __syncthreads();    

}

void int8quant_(int rows, int cols, const half * src, int8_t *output, 
        half *scale, cudaStream_t stream){


    dim3 block(256);
    dim3 grid(rows, 1);
    FindRowScaleKernel_<256, half><<<grid, block, 0, stream>>>(
                output,
                src, scale,
                rows, cols);

};

void int8quant_(int rows, int cols, const half * src, int8_t *output, 
        float *scale, cudaStream_t stream){


    dim3 block(256);
    dim3 grid(rows, 1);
    FindRowScaleKernel_<256,float><<<grid, block, 0, stream>>>(
                output,
                src, scale,
                rows, cols);

};




static constexpr int clamp(int val) {
    if (val < -128.0) 
        return 127.0;
    if (val > 127.0)
        return 127.0;
    return val;
}

template<int size, typename T>
__global__ void FindRowScaleKernel_cached_scaling(int8_t * output, const half * d_in,
         T * scale, int rows, int cols){


    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x ;
    if (bid > rows)
        return ;
    const  __half *start = d_in + bid * cols;
    int8_t * d_out = output + bid * cols;


    half quant_scales =  (__float2half) (scale[bid] );

    // quant
    for (int i = tid ; i < cols; i += size){
        int tmp =  __half2int_rn( __hdiv( start[i], quant_scales ) );
        d_out[i] =  static_cast<int8_t>(clamp(tmp))  ;
    } 
    __syncthreads();    

}
void int8quant_with_cached_scaling(int rows, int cols, const half * src, int8_t *output, 
        float *scale, cudaStream_t stream){

    dim3 block(256);
    dim3 grid(rows, 1);
    FindRowScaleKernel_cached_scaling<256,float><<<grid, block, 0, stream>>>(
                output,
                src, scale,
                rows, cols);

}


__global__  void FindOutliersAndSetToZeros_kernel_(const int *ind,  half *input, 
        half *outliersOutput, int m, int k, int len){
 

    int tid = threadIdx.x;
 
    int start_col = blockIdx.x ;
 
    if (start_col > len)
        return ;

  
 
 
    int col = ind[start_col];
    half *start = input +  col ;
    half *outliersOutput_ = outliersOutput + start_col;   
 
    for (int i = tid; i < m ; i+=  128  ){
        outliersOutput_[ i * len ] = start[ i * k ] ;
        // start[ i * k ] = 0.0;

    }
 
 


}
void ExtractOutliersAndSetToZeros_(int M, int N, const half * A, half *fp_A, 
        const int *ind, const int len, cudaStream_t stream){


    const int blockSize = 128;
 

    half * tmp = const_cast<half*>(A);
    dim3 numBlocks(len);        
    FindOutliersAndSetToZeros_kernel_<<<numBlocks, blockSize, 0, 
            stream>>>(
            ind,
            tmp,
            fp_A,
            M,
            N,
            len
        );

}

void gemmfp16_(
    const half * mat1,
    const half * mat2, half *mat3, int m, int n, int k, 
    cublasHandle_t handle, cudaStream_t stream)
     {
 

  static float _beta = 0.0;
  static  float _alpha = 1.0;

  auto beta_ptr = (void*)&_beta;
  auto alpha_ptr = (void*)&_alpha;

  auto input_ptr = (void*)mat3;
  auto mat1_ptr = (void*)mat1;
  auto mat2_ptr = (void*)mat2;

    
  (cublasGemmEx(
       handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      alpha_ptr,
      mat2_ptr,
      CUDA_R_16F,
      k,
      mat1_ptr,
      CUDA_R_16F,
      k,
      beta_ptr,
      input_ptr,
      CUDA_R_16F,
      n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  

};


#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_REPEAT_M 2
#define WARP_SIZE 32

__device__ inline void mma(const uint32_t* a, const uint32_t* b, uint32_t* frag_c) {
  float* c = reinterpret_cast<float*>(frag_c);
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b[0]),  "r"(b[1]),
       "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3])
  );
}


__global__ void mmaNaiveKernel(const half *__restrict__ A, 
                               const half *__restrict__ B, 
                               half *__restrict__ C, size_t M,
                               size_t N, size_t K) {
    const size_t K_tiles = div_ceil(K, MMA_K);

    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;

    if (warp_row >= M || warp_col >= N) {
        return;
    }

    __shared__ half A_smem[MMA_M][MMA_K];
    __shared__ half B_smem[MMA_N][MMA_K];
    __shared__ half C_smem[MMA_M][MMA_N];
    // __shared__ float D_smem[MMA_M][MMA_N];

    const size_t lane_id = threadIdx.x % WARP_SIZE;

    uint32_t RC[4] = {0, 0, 0, 0};
    uint32_t RD[2] = {0, 0};

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&A[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2);

        if (lane_id < MMA_N * 2) {
            *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =
                *((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
        }

        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];

        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);
        // if (lane_id == 0){
        //     half * tmp = reinterpret_cast<half *>(RA);
        //     for (int i = 0 ; i< 8; ++i)
        //         printf("ra %d = %.2f\t", i, __half2float(tmp[i]));
        //     printf("\n");
        //     tmp = reinterpret_cast<half *>(RB);
        //     for (int i = 0 ; i< 4; ++i)
        //         printf("rb %d = %.2f\t", i,  __half2float(tmp[i]));
        //     printf("\n");
        // }
        // HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
        mma(RA, RB, RC); // RC 4 float result  4 * 32  = 128 bit
        __syncthreads();
    }

    float *c_ = reinterpret_cast<float *>(RC);
    half *d_ = reinterpret_cast<half *>(RD);
    for (int i = 0 ; i < 4; ++i) {
        d_[i] = __float2half(c_[i]);
    }
    // convert to 

    *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RD[0];
    *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RD[1];

    __syncthreads();

    if (lane_id < MMA_M) {
        *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&C_smem[lane_id][0]));
    }
}




__global__ void mma_sparse_A_dense_B_kernel(const half *__restrict__ A, 
                               const half *__restrict__ B, 
                               half *__restrict__ C, size_t M,
                               size_t N, size_t K) {
    const size_t K_tiles = 1;
    assert ( K <= MMA_K);

    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;

    if (warp_row >= M || warp_col >= N) {
        return;
    }

    // __shared__ half A_smem[MMA_M][MMA_K];
    // __shared__ half B_smem[MMA_N][MMA_K];
    __shared__ half C_smem[MMA_M][MMA_N];
    // __shared__ float D_smem[MMA_M][MMA_N];

    const size_t lane_id = threadIdx.x % WARP_SIZE;

    uint32_t RC[4] = {0, 0, 0, 0};
    uint32_t RD[2] = {0, 0};

 
    // *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
    //     *((int4 *)(&A[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2);

    // if (lane_id < MMA_N * 2) {
    //     *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =
    //         *((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
    // }

    // __syncthreads();

    uint32_t RA[4] = {0, 0, 0, 0};
    // 每一个线程需要 32 * 4 / 16  = 8 个half
    uint32_t RB[2] = {0, 0};

    // const int4 * A_ =  reinterpret_cast<const int4 *>(A);
    const uint32_t * B_ =  reinterpret_cast<const uint32_t *>(B); 
    const uint32_t * A_ =  reinterpret_cast<const uint32_t *>(A); 

    if (K == 16){
        // A_ += warp_row * K / 8 +  (lane_id / 2 )  * K / 8 +   ( (lane_id % 2 ));
        // RA[0] = A_[0];

        // B = B +  (warp_col + lane_id / 4 ) * K +  (lane_id % 4) * 4;
        // half * B_ =  reinterpret_cast<half *>(RB);
        // for (int i = 0; i < 4; ++i){
        //     RB[i] = B_[i];
        // }
        RA[0] =  *( A_  + (warp_row + lane_id / 4 ) * (K/2)      +  (lane_id % 4) );
        RA[1] =  *( A_  + (warp_row + lane_id / 4  + 8) * (K/2)  +  (lane_id % 4) );
        RA[2] =  *( A_  + (warp_row + lane_id / 4 ) * (K/2)      +  (lane_id % 4) + 4);
        RA[3] =  *( A_  + (warp_row + lane_id / 4  + 8) * (K/2)  +  (lane_id % 4) + 4);

        RB[0] =  *( B_  + (warp_col + lane_id / 4 ) * (K/2)  +  (lane_id % 4) );
        RB[1] =  *( B_  + (warp_col + lane_id / 4 ) * (K/2)  +  (lane_id % 4)  + 4 );
    
    }
    
    if (K == 8){
        // A_ += warp_row * K / 8 +  (lane_id / 2 )  * K / 8 +   ( (lane_id % 2 ));
        // RA[0] = A_[0];

        // B = B +  (warp_col + lane_id / 4 ) * K +  (lane_id % 4) * 4;
        // half * B_ =  reinterpret_cast<half *>(RB);
        // for (int i = 0; i < 4; ++i){
        //     RB[i] = B_[i];
        // }
        RA[0] =  *( A_  + (warp_row + lane_id / 4 ) * (K/2)      +  (lane_id % 4) );
        RA[1] =  *( A_  + (warp_row + lane_id / 4  + 8) * (K/2)  +  (lane_id % 4) );
        // RA[2] =  *( A_  + (warp_row + lane_id / 4 ) * (K/2)      +  (lane_id % 4) + 4);
        // RA[3] =  *( A_  + (warp_row + lane_id / 4  + 8) * (K/2)  +  (lane_id % 4) + 4);

        RB[0] =  *( B_  + (warp_col + lane_id / 4 ) * (K/2)  +  (lane_id % 4) );
        // RB[1] =  *( B_  + (warp_col + lane_id / 4 ) * (K/2)  +  (lane_id % 4)  + 4 );
    
    }
    // if (lane_id == 0){
    //     half * tmp = reinterpret_cast<half *>(RA);
    //     for (int i = 0 ; i< 8; ++i)
    //         printf("ra %d = %.2f\t", i, __half2float(tmp[i]));
    //     printf("\n");
    //     tmp = reinterpret_cast<half *>(RB);
    //     for (int i = 0 ; i< 4; ++i)
    //         printf("rb %d = %.2f\t", i, __half2float(tmp[i]));
    //     printf("\n");
    // }
    // uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
    // LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

    // uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
    // LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

    // HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
    __syncthreads();
    mma(RA, RB, RC); // RC 4 float result  4 * 32  = 128 bit
    __syncthreads();


    float *c_ = reinterpret_cast<float *>(RC);
    half *d_ = reinterpret_cast<half *>(RD);
    for (int i = 0 ; i < 4; ++i) {
        d_[i] = __float2half(c_[i]);
    }
    // convert to 

    *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RD[0];
    *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RD[1];

    __syncthreads();

    if (lane_id < MMA_M) {
        *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&C_smem[lane_id][0]));
    }
}

void mixgemm_sparse_fp16_dense_weight_cu(int M, int N, int K, int num_ind,
                                        half* A, half* B, half* out, 
                                        int* ind, cudaStream_t stream){

    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));

    assert ((N % MMA_N ) == 0);
    assert ((M % MMA_M ) == 0);
    assert ((M >= MMA_M ) == 0);
    assert ((N >= MMA_N ) == 0);


    size_t smem_max_size =  2 * ((MMA_M * MMA_K) + (MMA_N * MMA_K) + (MMA_M * MMA_N)) * sizeof(half);
    cudaFuncSetAttribute(mma_sparse_A_dense_B_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size);
    mma_sparse_A_dense_B_kernel<<<grid, block, smem_max_size, stream>>>(A, B, out, M, N, K);
    // mmaNaiveKernel<<<grid, block, smem_max_size, stream>>>(A, B, out, M, N, K);

}


__global__ void zero_copy_kernel(int *flag,  half* input){

    if (threadIdx.x == 0)
    if (__half2float(input[0]) > 1.00){
        flag[0] = 1;
    }
}
void zero_copy_cu(int *flag,  half* input){

    dim3 block(32);
    dim3 grid(1, 1);
    zero_copy_kernel<<<grid, block, 0>>>(flag, input);

}

__global__ void find_zeros_kernel(int * dev_ptr, half* input, int bs, int seq, int hidden, half* last_input){
    int tid  = threadIdx.x;

    float tmp = __half2float(input[0]);
    if (tmp == 0.0){
        if (tid == 0)
            dev_ptr[0] = 1;

        #pragma unroll
        for (int i = 0 ; i < 8; ++i)
            last_input[tid * 8 + i] = input[  ( seq + tid ) * hidden + i];

    }

}
void find_zeros_cu(int * dev_ptr, half* input, int bs, int seq, int hidden, half* last_input){

    dim3 block(32); //thread
    dim3 grid(1, 1); 

    find_zeros_kernel<<<grid, block, 0>>>(dev_ptr, input, bs, seq, hidden, last_input);
}



__global__ void reuse_output_kernel(int * dev_ptr, half* input, int bs, int seq, int hidden, half* last_input){
    
    int tid  = threadIdx.x;
    float local_result = 0.0;

    #pragma unroll
    for (int i = 0 ; i < 8; ++i){
        
        local_result +=  abs( __half2float(last_input[tid * 8 + i])  - __half2float(input[  ( seq + tid ) * hidden + i]) ) ;
    }

    local_result = warpReduceSum(local_result, blockDim.x);
     
    if (tid == 0)
        if (local_result == 0.0){
        
            dev_ptr[0] = 1;
    }

}

void reuse_output_cu(int * dev_ptr, half* input, int bs, int seq, int hidden, half* last_input){

    dim3 block(32); //thread
    dim3 grid(1, 1); 

    reuse_output_kernel<<<grid, block, 0>>>(dev_ptr, input, bs, seq, hidden, last_input);
}