

#include "kernel.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <assert.h>

void check(cudaError_t result, char const* const func, const char* const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error = %s at %s:%d '%s'\n",
            cudaGetErrorString(result), file, line, func);
    exit(1);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#define WARP_SIZE 32
#define SHARED_MEM_MAX_ROWS 64
#define MAX_THREADS_PER_BLOCK 1024





template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

using FragB = Vec<half2, 2>;


template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}
__device__ inline FragB dequant(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  FragB frag_b;
  frag_b[0] = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  frag_b[1] = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
  );
  return frag_b;
}


using FragD = Vec<half, 8>;
__device__ inline FragD dequantLong(int source){


        FragD result;
       //	return result;
        uint32_t* h = reinterpret_cast<uint32_t*>(&result);
        uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

        // First, we extract the i4s and construct an intermediate fp16 number.
        static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
        static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
        static constexpr uint32_t TOP_MASK = 0x00f000f0;
        static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

        // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
        // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
        // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
        // elt_67 to fp16 without having to shift them to the bottom bits before hand.

        // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
        // immediately before required.
        const uint32_t top_i4s = i4s >> 8;
        // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[0])
                     : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[1])
                     : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[2])
                     : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[3])
                     : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

        // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
        // half2 ctor. In this case, I chose performance reliability over code readability.

        // This is the half2 {1032, 1032} represented as an integer.
        static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
        // This is the half2 {1 / 16, 1 / 16} represented as an integer.
        static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
        // This is the half2 {-72, -72} represented as an integer.
        static constexpr uint32_t NEG_72 = 0xd480d480;

        // Finally, we construct the output numbers.
        // Convert elt_01
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
        // Convert elt_23
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
        // Convert elt_45
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
        // Convert elt_67
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_72));

        return result;

}





 

template<int size>
__global__ void FindRowScaleKernel(int8_t * output, 
            const __nv_bfloat16 * d_in, 
            float * scale, 
            int rows, int cols){

    __shared__ __nv_bfloat16 sdata[size];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x ;
    if (bid > rows)
        return ;
    const  __nv_bfloat16 *start = d_in + bid * cols;
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

    float  max = __bfloat162float(sdata[0]);
    __syncthreads();

    float quant_scales = ( max / 127.0);
    if (tid == 0){
        scale[bid] = quant_scales;
    }
    // quant
    for (int i = tid ; i < cols; i += size)
        d_out[i] =  static_cast<int8_t>(__float2int_rn(  __bfloat162float(start[i]) / quant_scales  ))  ; 
    __syncthreads();    

}

template<int size>
__global__ void FindRowScaleKernel(int8_t * output, 
            const float * d_in, 
            float * scale, 
            int rows, int cols){

    __shared__ float sdata[size];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x ;
    if (bid > rows)
        return ;
    const  float *start = d_in + bid * cols;
    int8_t * d_out = output + bid * cols;
    sdata[tid] = abs(start[tid]); 
    for (int i = tid + size; i < cols; i += size)
        sdata[tid] = max ( abs(start[i]),  sdata[tid] ); 
    __syncthreads();


    // do reduction in shared mem
    for (unsigned int s= blockDim.x/2; s >= 1; s >>=1 ) {
        if (tid < s) {
            sdata[tid] =  max ( abs(sdata[tid + s]),  sdata[tid]);
        }
        __syncthreads();
    }

    float  max_ =  (sdata[0]);
    __syncthreads();

    float quant_scales = ( max_ / 127.0);
    if (tid == 0){
        scale[bid] = quant_scales;
    }
    // quant
    for (int i = tid ; i < cols; i += size)
        d_out[i] =  static_cast<int8_t>(__float2int_rn(   (start[i]) / quant_scales  ))  ; 
    __syncthreads();    

}


void FindRowScaleFloat(  float *x,   float *scaleRow,
                         int rows, int cols, int8_t * out, cudaStream_t stream){



    // auto options = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    // auto quant_out = torch::zeros(
    //     {rows, cols}, options);
        dim3 block(256);
        dim3 grid(rows, 1);

    FindRowScaleKernel<256><<<grid, block, 1024, stream>>>(
        out,
        x, 
        scaleRow,
        rows, cols);
    // return quant_out;
 

}

void FindRowScaleBF16(  __nv_bfloat16 *x,   float *scaleRow,
                         int rows, int cols , int8_t * out, cudaStream_t stream){

    dim3 block(256);
    dim3 grid(rows, 1);

    FindRowScaleKernel<256><<<grid, block, 1024, stream>>>(
        out,
        x, 
        scaleRow,
        rows, cols);
    // return quant_out;


 }


 __global__ void gemv_int4_fp16_kernel_sm90(int32_t* mat, half* vec, half* res, unsigned int k_reduction,
                          unsigned int num_per_thread, float * scaling,
                          half* weight, int* ind, int n_outliers, 
                          unsigned int num_outliers_per_thread) {
  float sum = 0;
  float sum_2 = 0.0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;

  unsigned int row = ( blockIdx.y * blockDim.y   + threadIdx.y) * 2 ;
  unsigned int start_idx = threadIdx.x;
  int* mat4 = reinterpret_cast<int*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);


#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < k_reduction >> 3) {
      float4 vec_val = vec4[j];
    //   float4 mat_val = mat4[ row * (  (n >> 3) / 4 ) + j / 4 ];  

      int *read =  mat4 +  row * (  (k_reduction  >> 3)   ) + j * 2   ;
      int* read_int = reinterpret_cast<int*>(read);

      int b_quant = read_int[0];
      int b_quant2 = read_int[1];
      
      
      // 一次读一个int32 可能打不满H100的带宽
      // 尝试一次读一个int64
      int b_quant_shift = b_quant >> 8;
      int b_quant_shift2 = b_quant2 >> 8;
      FragB frag_b0 = dequant(b_quant);
      FragB frag_b1 = dequant(b_quant_shift);


      FragB frag_b0_2 = dequant(b_quant2);
      FragB frag_b1_2 = dequant(b_quant_shift2);


       half2* vec_h1 = (half2*)&vec_val.x;
       half2* vec_h2 = (half2*)&vec_val.y;
       half2* vec_h3 = (half2*)&vec_val.z;
       half2* vec_h4 = (half2*)&vec_val.w;

       half2* mat_h1 = (half2*)&frag_b0[0];
       half2* mat_h2 = (half2*)&frag_b0[1];
       half2* mat_h3 = (half2*)&frag_b1[0];
       half2* mat_h4 = (half2*)&frag_b1[1];

       half2* mat_h1_2 = (half2*)&frag_b0_2[0];
       half2* mat_h2_2 = (half2*)&frag_b0_2[1];
       half2* mat_h3_2 = (half2*)&frag_b1_2[0];
       half2* mat_h4_2 = (half2*)&frag_b1_2[1];

      sum += __half2float(vec_h1->x) * __half2float(mat_h1->x);
      sum += __half2float(vec_h1->y) * __half2float(mat_h1->y);
      sum += __half2float(vec_h2->x) * __half2float(mat_h2->x);
      sum += __half2float(vec_h2->y) * __half2float(mat_h2->y);
      sum += __half2float(vec_h3->x) * __half2float(mat_h3->x);
      sum += __half2float(vec_h3->y) * __half2float(mat_h3->y);
      sum += __half2float(vec_h4->x) * __half2float(mat_h4->x);
      sum += __half2float(vec_h4->y) * __half2float(mat_h4->y);

      sum_2 += __half2float(vec_h1->x) * __half2float(mat_h1_2->x);
      sum_2 += __half2float(vec_h1->y) * __half2float(mat_h1_2->y);
      sum_2 += __half2float(vec_h2->x) * __half2float(mat_h2_2->x);
      sum_2 += __half2float(vec_h2->y) * __half2float(mat_h2_2->y);
      sum_2 += __half2float(vec_h3->x) * __half2float(mat_h3_2->x);
      sum_2 += __half2float(vec_h3->y) * __half2float(mat_h3_2->y);
      sum_2 += __half2float(vec_h4->x) * __half2float(mat_h4_2->x);
      sum_2 += __half2float(vec_h4->y) * __half2float(mat_h4_2->y);
      
    }
  }

  // 计算outliers
  float* mat2_weight_fp16 = reinterpret_cast<float*>(weight);
  // half* vec2_outliers = reinterpret_cast<half*>(vec);
  // int * ind_outlier =  ind;

#pragma unroll
  for (int iter = 0; iter < num_outliers_per_thread >> 1; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n_outliers >> 1) {
      // float4 vec_val = vec4[j];
      float * mat_val = mat2_weight_fp16 +  row * (n_outliers >> 1) + j * 2;
      half2* read_mat = reinterpret_cast<half2*>(mat_val);

      int ind1 = ind[ 2 * j];
      int ind2 = ind[ 2 * j + 1];


      half vec_h1 = vec[ind1];
      half vec_h2 = vec[ind2];

        
      half2 mat_h1 = read_mat[0];
      half2 mat_h1_2 = read_mat[1];
     
      sum += __half2float(vec_h1) * __half2float(mat_h1.x);
      sum += __half2float(vec_h2) * __half2float(mat_h1.y);
      sum_2 += __half2float(vec_h1) * __half2float(mat_h1_2.x);
      sum_2 += __half2float(vec_h2) * __half2float(mat_h1_2.y);   
    }
  }

  sum = warpReduceSum(sum, blockDim.x);
  sum_2 = warpReduceSum(sum_2, blockDim.x);

 


  
  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2half(sum  * scaling[row]);
      res[row+1] = __float2half(sum_2  * scaling[row + 1]);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  static __shared__ float warpLevelSums_2[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) {
      warpLevelSums[threadIdx.y][warpId] = sum;
      warpLevelSums_2[threadIdx.y][warpId] = sum_2;
  }
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;

  sum_2 = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums_2[threadIdx.y][laneId]
            : 0.0;

  // Final reduce using first warp
  if (warpId == 0) {
      sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
      sum_2 = warpReduceSum(sum_2, blockDim.x / WARP_SIZE);
  }
 
  if (tid == 0) {
      res[row] = __float2half(sum  * scaling[row]);
      res[row+1] = __float2half(sum_2  * scaling[row + 1]);
  }
}

 


__global__ void gemv_fp16(half* mat, half* vec, half* res, unsigned int n,
                          unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  float4* mat4 = reinterpret_cast<float4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      float4 vec_val = vec4[j];
      float4 mat_val = mat4[row * (n >> 3) + j];
       half2* vec_h1 = (half2*)&vec_val.x;
       half2* vec_h2 = (half2*)&vec_val.y;
       half2* vec_h3 = (half2*)&vec_val.z;
       half2* vec_h4 = (half2*)&vec_val.w;
       half2* mat_h1 = (half2*)&mat_val.x;
       half2* mat_h2 = (half2*)&mat_val.y;
       half2* mat_h3 = (half2*)&mat_val.z;
       half2* mat_h4 = (half2*)&mat_val.w;
      sum += __half2float(vec_h1->x) * __half2float(mat_h1->x);
      sum += __half2float(vec_h1->y) * __half2float(mat_h1->y);
      sum += __half2float(vec_h2->x) * __half2float(mat_h2->x);
      sum += __half2float(vec_h2->y) * __half2float(mat_h2->y);
      sum += __half2float(vec_h3->x) * __half2float(mat_h3->x);
      sum += __half2float(vec_h3->y) * __half2float(mat_h3->y);
      sum += __half2float(vec_h4->x) * __half2float(mat_h4->x);
      sum += __half2float(vec_h4->y) * __half2float(mat_h4->y);
    }
  }

  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2half(sum);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2half(sum);
  }
}


__global__ void gemv_int4_kernel(int32_t* mat, half* vec, half* res, unsigned int n,
                          unsigned int num_per_thread, float * scaling) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  int* mat4 = reinterpret_cast<int*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      float4 vec_val = vec4[j];
    //   float4 mat_val = mat4[ row * (  (n >> 3) / 4 ) + j / 4 ]; 

      int b_quant = mat4[ row * (  (n >> 3)   ) + j   ]; 
      int b_quant_shift = b_quant >> 8;
      FragB frag_b0 = dequant(b_quant);
      FragB frag_b1 = dequant(b_quant_shift);

       half2* vec_h1 = (half2*)&vec_val.x;
       half2* vec_h2 = (half2*)&vec_val.y;
       half2* vec_h3 = (half2*)&vec_val.z;
       half2* vec_h4 = (half2*)&vec_val.w;

       half2* mat_h1 = (half2*)&frag_b0[0];
       half2* mat_h2 = (half2*)&frag_b0[1];
       half2* mat_h3 = (half2*)&frag_b1[0];
       half2* mat_h4 = (half2*)&frag_b1[1];
      sum += __half2float(vec_h1->x) * __half2float(mat_h1->x);
      sum += __half2float(vec_h1->y) * __half2float(mat_h1->y);
      sum += __half2float(vec_h2->x) * __half2float(mat_h2->x);
      sum += __half2float(vec_h2->y) * __half2float(mat_h2->y);
      sum += __half2float(vec_h3->x) * __half2float(mat_h3->x);
      sum += __half2float(vec_h3->y) * __half2float(mat_h3->y);
      sum += __half2float(vec_h4->x) * __half2float(mat_h4->x);
      sum += __half2float(vec_h4->y) * __half2float(mat_h4->y);
    }
  }

  sum = warpReduceSum(sum, blockDim.x);

  
  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2half(sum  * scaling[row]);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2half(sum);
  }
}



__global__ void dequant_int4_kernel(int32_t* mat,  half* res, unsigned int reduction_dim,
                          unsigned int num_per_thread, float * scaling, int groupsize) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  int* mat4 = reinterpret_cast<int*>(mat); 

  int scale_num_each_n =  reduction_dim / groupsize;
#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < reduction_dim >> 3) {
       half * out = res + row * reduction_dim + j * 8;

      int b_quant = mat4[ row * (  (reduction_dim >> 3)   ) + j   ]; 
      int b_quant_shift = b_quant >> 8;
      FragB frag_b0 = dequant(b_quant);
      FragB frag_b1 = dequant(b_quant_shift);
 
       half2* mat_h1 = (half2*)&frag_b0[0];
       half2* mat_h2 = (half2*)&frag_b0[1];
       half2* mat_h3 = (half2*)&frag_b1[0];
       half2* mat_h4 = (half2*)&frag_b1[1];

      out[0] =  __half2float(mat_h1->x) *  scaling[row * scale_num_each_n  + (j * 8) / groupsize];
      out[1] =  __half2float(mat_h1->y) *  scaling[row * scale_num_each_n  + (j * 8) / groupsize];
      out[2] =  __half2float(mat_h2->x) *  scaling[row * scale_num_each_n  + (j * 8) / groupsize];
      out[3] =  __half2float(mat_h2->y) *  scaling[row * scale_num_each_n  + (j * 8) / groupsize];
      out[4] =  __half2float(mat_h3->x) *  scaling[row * scale_num_each_n  + (j * 8) / groupsize];
      out[5] =  __half2float(mat_h3->y) *  scaling[row * scale_num_each_n  + (j * 8) / groupsize];
      out[6] =  __half2float(mat_h4->x) *  scaling[row * scale_num_each_n  + (j * 8) / groupsize];
      out[7] =  __half2float(mat_h4->y) *  scaling[row * scale_num_each_n  + (j * 8) / groupsize];

    }
  }

}

__global__ void gemv_int4_kernel_sm90(int32_t* mat, half* vec, half* res, unsigned int k_reduction,
                          unsigned int num_per_thread, float * scaling) {
  float sum = 0;
  float sum_2 = 0.0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  // block idx y 
  // threadIdx.y 0 ~ 4
  // blockDim.y  4
  // blockiIdx y 0 1 2 1024
  // thread id y 0 1 2 3
  // n = 4096
  //   n
  // n/4
  // 4
  // 例如 n = 24
  // (id) * 4 + 4
  // id = 0 1 2 3 4 5 
  // 现在id = 0 1 2
  // (id) * 4 * 2 + 4 * 2
  // 2 * 4 * 2 + 8
  // 2 * 4 * 2 + 6


  unsigned int row = ( blockIdx.y * blockDim.y   + threadIdx.y) * 2 ;
  unsigned int start_idx = threadIdx.x;
  int* mat4 = reinterpret_cast<int*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);


#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < k_reduction >> 3) {
      float4 vec_val = vec4[j];
    //   float4 mat_val = mat4[ row * (  (n >> 3) / 4 ) + j / 4 ];  

      int *read =  mat4 +  row * (  (k_reduction  >> 3)   ) + j * 2   ;
      int* read_int = reinterpret_cast<int*>(read);

      int b_quant = read_int[0];
      int b_quant2 = read_int[1];
      
      
      // 一次读一个int32 可能打不满H100的带宽
      // 尝试一次读一个int64

      FragD frag_b0 = dequantLong(b_quant);
      FragD frag_b2 = dequantLong(b_quant2);


       half2* vec_h1 = (half2*)&vec_val.x;
       half2* vec_h2 = (half2*)&vec_val.y;
       half2* vec_h3 = (half2*)&vec_val.z;
       half2* vec_h4 = (half2*)&vec_val.w;



      sum += __half2float(vec_h1->x) * __half2float(frag_b0[0]);
      sum += __half2float(vec_h1->y) * __half2float(frag_b0[1]);
      sum += __half2float(vec_h2->x) * __half2float(frag_b0[2]);
      sum += __half2float(vec_h2->y) * __half2float(frag_b0[3]);
      sum += __half2float(vec_h3->x) * __half2float(frag_b0[4]);
      sum += __half2float(vec_h3->y) * __half2float(frag_b0[5]);
      sum += __half2float(vec_h4->x) * __half2float(frag_b0[6]);
      sum += __half2float(vec_h4->y) * __half2float(frag_b0[7]);

      sum_2 += __half2float(vec_h1->x) * __half2float(frag_b2[0]);
      sum_2 += __half2float(vec_h1->y) * __half2float(frag_b2[1]);
      sum_2 += __half2float(vec_h2->x) * __half2float(frag_b2[2]);
      sum_2 += __half2float(vec_h2->y) * __half2float(frag_b2[3]);
      sum_2 += __half2float(vec_h3->x) * __half2float(frag_b2[4]);
      sum_2 += __half2float(vec_h3->y) * __half2float(frag_b2[5]);
      sum_2 += __half2float(vec_h4->x) * __half2float(frag_b2[6]);
      sum_2 += __half2float(vec_h4->y) * __half2float(frag_b2[7]);
      
    }
  }

  sum = warpReduceSum(sum, blockDim.x);
  sum_2 = warpReduceSum(sum_2, blockDim.x);

  
  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2half(sum  * scaling[row]);
      res[row+1] = __float2half(sum_2  * scaling[row + 1]);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  static __shared__ float warpLevelSums_2[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) {
      warpLevelSums[threadIdx.y][warpId] = sum;
      warpLevelSums_2[threadIdx.y][warpId] = sum_2;
  }
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;

  sum_2 = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums_2[threadIdx.y][laneId]
            : 0.0;

  // Final reduce using first warp
  if (warpId == 0) {
      sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
      sum_2 = warpReduceSum(sum_2, blockDim.x / WARP_SIZE);
  }
 
  if (tid == 0) {
      res[row] = __float2half(sum  * scaling[row]);
      res[row+1] = __float2half(sum_2  * scaling[row + 1]);
  }
}



void gemv_cu(int m, int n, int k,  half * vec_data_,
                                  half * mat_data_, 
                                  half * result_data_,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y , cudaStream_t stream){
                      
 
  assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
  assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
  unsigned int num_per_thread = k / block_dim_x; 
  assert(num_per_thread >= 8);


//   half * mat_data_ = reinterpret_cast<half*>(weight.data_ptr<at::Half>());
//   half * vec_data_ = reinterpret_cast<half*>(x.data_ptr<at::Half>());
//   half * result_data_ = reinterpret_cast<half*>(_out.data_ptr<at::Half>());

  dim3 grid_dim(1, n / block_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  gemv_fp16<<<grid_dim, block_dim, 1024, stream>>>(mat_data_, vec_data_, result_data_,
                                     k, num_per_thread);
  checkCudaErrors(cudaPeekAtLastError());
  return  ;
}


void gemv_int4_cu(int m, int n, int k, half * vec_data_,
                                  void * mat_data_, 
                                 half * result_data_,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                    float * scaling_data_,
                                    cudaStream_t stream
                                   ){
                      
 
  assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
  assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
  unsigned int num_per_thread = k / block_dim_x; 
  // 对于int4 block dim 应该减半
  

  assert(num_per_thread >= 8);

//   half * vec_data_ = reinterpret_cast<half*>(x.data_ptr<at::Half>());
//   void * mat_data_ = (weight.data_ptr());
  
//   half * result_data_ = reinterpret_cast<half*>(_out.data_ptr<at::Half>());
//   float * scaling_data_ = reinterpret_cast<float*>(scaling.data_ptr<float>());


  dim3 grid_dim(1, n / block_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  gemv_int4_kernel<<<grid_dim, block_dim, 0, stream>>>((int32_t*) mat_data_, vec_data_, result_data_,
                                     k, num_per_thread, scaling_data_);
  checkCudaErrors(cudaPeekAtLastError());
  return  ;
}



void dequant_int4_cu( int n, int k,   void * mat_data_, 
                                 half * result_data_,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                    float * scaling_data_,
                                    cudaStream_t stream, 
                                    int groupsize
                                   ){
                      
 
  assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
  assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
  unsigned int num_per_thread = k / block_dim_x; 

  assert(num_per_thread >= 8);


  dim3 grid_dim(1, n / block_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  dequant_int4_kernel<<<grid_dim, block_dim, 0, stream>>>((int32_t*) mat_data_, result_data_,
                                     k, num_per_thread, scaling_data_, groupsize);
  checkCudaErrors(cudaPeekAtLastError());
  return  ;
}

void gemv_int4_sm90_cu(int m, int n, int k, half * vec_data_,
                                  void * mat_data_, 
                                  half * result_data_,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                   float * scaling_data_,
                                   cudaStream_t stream
                                   ){
                      
 
  assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
  assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
  unsigned int num_per_thread = k / block_dim_x; 
  // 对于int4 block dim 应该减半
  

  assert(num_per_thread >= 8);


//   void * mat_data_ = (weight.data_ptr());
//   half * vec_data_ = reinterpret_cast<half*>(x.data_ptr<at::Half>());
//   half * result_data_ = reinterpret_cast<half*>(_out.data_ptr<at::Half>());
//   float * scaling_data_ = reinterpret_cast<float*>(scaling.data_ptr<float>());


  dim3 grid_dim(1, (n / 2) / block_dim_y);
  // 需要的 block的数量减少一半
  dim3 block_dim(block_dim_x, block_dim_y);
  gemv_int4_kernel_sm90<<<grid_dim, block_dim, 1024, stream>>>((int32_t*) mat_data_, vec_data_, result_data_,
                                     k, num_per_thread, scaling_data_);
  checkCudaErrors(cudaPeekAtLastError());
  return  ;
}


__global__ void gemv_fp16_int4_mix(int32_t* mat, half* vec, half* res, unsigned int n,
                          unsigned int num_per_thread, float * scaling,
                          half* weight, int* ind, int n_outliers, 
                          unsigned int num_outliers_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  int* mat4 = reinterpret_cast<int*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {

      
        float4 vec_val = vec4[j];
      //   float4 mat_val = mat4[ row * (  (n >> 3) / 4 ) + j / 4 ];  
        int b_quant = __ldg(mat4 + row * (  (n >> 3)   ) + j   ); 
        int b_quant_shift = b_quant >> 8;



       FragB frag_b0 = dequant(b_quant);
       FragB frag_b1 = dequant(b_quant_shift);

       half2* vec_h1 = (half2*)&vec_val.x;
       half2* vec_h2 = (half2*)&vec_val.y;
       half2* vec_h3 = (half2*)&vec_val.z;
       half2* vec_h4 = (half2*)&vec_val.w;

       half2* mat_h1 = (half2*)&frag_b0[0];
       half2* mat_h2 = (half2*)&frag_b0[1];
       half2* mat_h3 = (half2*)&frag_b1[0];
       half2* mat_h4 = (half2*)&frag_b1[1];
      sum += __half2float(vec_h1->x) * __half2float(mat_h1->x);
      sum += __half2float(vec_h1->y) * __half2float(mat_h1->y);
      sum += __half2float(vec_h2->x) * __half2float(mat_h2->x);
      sum += __half2float(vec_h2->y) * __half2float(mat_h2->y);
      sum += __half2float(vec_h3->x) * __half2float(mat_h3->x);
      sum += __half2float(vec_h3->y) * __half2float(mat_h3->y);
      sum += __half2float(vec_h4->x) * __half2float(mat_h4->x);
      sum += __half2float(vec_h4->y) * __half2float(mat_h4->y);
    }
  }

  


  // 原来是float4 8 个half
  // outliers是 float 2 个half
  // 每个线程的计算量
  // 比如 outlier 是 128的时候
  // blockdim x 是64的时候，每个线程的计算量就是2
  float* mat2_weight_fp16 = reinterpret_cast<float*>(weight);
  // half* vec2_outliers = reinterpret_cast<half*>(vec);
  // int * ind_outlier =  ind;

#pragma unroll
  for (int iter = 0; iter < num_outliers_per_thread >> 1; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n_outliers >> 1) {
      // float4 vec_val = vec4[j];
      float mat_val = mat2_weight_fp16[row * (n_outliers >> 1) + j];
      int ind1 = ind[ 2 * j];
      int ind2 = ind[ 2 * j + 1];


      half vec_h1 = vec[ind1];
      half vec_h2 = vec[ind2];

        
      half2* mat_h1 = (half2*)&mat_val;
     
      sum += __half2float(vec_h1) * __half2float(mat_h1->x);
      sum += __half2float(vec_h2) * __half2float(mat_h1->y);
   
    }
  }


  sum = warpReduceSum(sum, blockDim.x);

  
  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2half(sum  * scaling[row]);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2half(sum  * scaling[row]);
  }
}


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
                                   ){
                      
 
  assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
  assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
  unsigned int num_per_thread = k / block_dim_x; 

  unsigned int num_outliers_per_thread = n_outliers / block_dim_x; 
  // 对于int4 block dim 应该减半
  

  assert(num_per_thread >= 8);



//   void * mat_data_ = (weight.data_ptr());
//   half * vec_data_ = reinterpret_cast<half*>(x.data_ptr<at::Half>());
//   half * result_data_ = reinterpret_cast<half*>(_out.data_ptr<at::Half>());
//   float * scaling_data_ = reinterpret_cast<float*>(scaling.data_ptr<float>());

//   half * weight_cache_data = reinterpret_cast<half*>(weight_cache.data_ptr<at::Half>());
//   int * ind_data = reinterpret_cast<int*>(ind.data_ptr<int>());



  dim3 grid_dim(1, n / block_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  gemv_fp16_int4_mix<<<grid_dim, block_dim, 1024, stream>>>((int32_t*) mat_data_, vec_data_, result_data_,
                                     k, num_per_thread, 
                                     scaling_data_,
                                     weight_cache_data,
                                     ind_data,
                                     n_outliers,
                                     num_outliers_per_thread);
  checkCudaErrors(cudaPeekAtLastError());
  return  ;
}




// add outliers
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
                                   ){
                      
 
  assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
  assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
  unsigned int num_per_thread = k / block_dim_x; 
  unsigned int num_outliers_per_thread = n_outliers / block_dim_x; 
  // 对于int4 block dim 应该减半
  assert(num_per_thread >= 8);
  dim3 grid_dim(1, (n / 2) / block_dim_y);
  // 需要的 block的数量减少一半
  dim3 block_dim(block_dim_x, block_dim_y);
  gemv_int4_fp16_kernel_sm90<<<grid_dim, block_dim, 1024, stream>>>((int32_t*) mat_data_, vec_data_, result_data_,
                                     k, num_per_thread, scaling_data_,
                                     weight_cache_data, ind_data, n_outliers, num_outliers_per_thread);
  checkCudaErrors(cudaPeekAtLastError());
  return  ;
}






__global__ void gemv_int4_fp16_kernel_sm90_new(int32_t* mat, half* vec, half* res, unsigned int k_reduction,
                          unsigned int num_per_thread, float * scaling,
                          half* weight, int* ind, int n_outliers, 
                          unsigned int num_outliers_per_thread) {
  float sum = 0;
  float sum_2 = 0.0;

  float sum_3 = 0;
  float sum_4 = 0.0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;

  unsigned int row = ( blockIdx.y * blockDim.y   + threadIdx.y) * 4 ;
  unsigned int start_idx = threadIdx.x;

  
  int* mat4 = reinterpret_cast<int*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);



#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < k_reduction >> 3) {
      float4 vec_val = vec4[j];

      int *read =  mat4 +  row * (  (k_reduction  >> 3)   ) + j * 4   ;
      int* read_int = reinterpret_cast<int*>(read);

      int b_quant = read_int[0];
      int b_quant2 = read_int[1];
      int b_quant3 = read_int[2];
      int b_quant4 = read_int[3];      
      

      int b_quant_shift = b_quant >> 8;
      int b_quant_shift2 = b_quant2 >> 8;
      int b_quant_shift3 = b_quant3 >> 8;
      int b_quant_shift4 = b_quant4 >> 8;



      FragB frag_b0 = dequant(b_quant);
      FragB frag_b1 = dequant(b_quant_shift);


      FragB frag_b0_2 = dequant(b_quant2);
      FragB frag_b1_2 = dequant(b_quant_shift2);

      FragB frag_b0_3 = dequant(b_quant3);
      FragB frag_b1_3 = dequant(b_quant_shift3);

      FragB frag_b0_4 = dequant(b_quant4);
      FragB frag_b1_4 = dequant(b_quant_shift4);

       half2* vec_h1 = (half2*)&vec_val.x;
       half2* vec_h2 = (half2*)&vec_val.y;
       half2* vec_h3 = (half2*)&vec_val.z;
       half2* vec_h4 = (half2*)&vec_val.w;

       half2* mat_h1 = (half2*)&frag_b0[0];
       half2* mat_h2 = (half2*)&frag_b0[1];
       half2* mat_h3 = (half2*)&frag_b1[0];
       half2* mat_h4 = (half2*)&frag_b1[1];

       half2* mat_h1_2 = (half2*)&frag_b0_2[0];
       half2* mat_h2_2 = (half2*)&frag_b0_2[1];
       half2* mat_h3_2 = (half2*)&frag_b1_2[0];
       half2* mat_h4_2 = (half2*)&frag_b1_2[1];

       half2* mat_h1_3 = (half2*)&frag_b0_3[0];
       half2* mat_h2_3 = (half2*)&frag_b0_3[1];
       half2* mat_h3_3 = (half2*)&frag_b1_3[0];
       half2* mat_h4_3 = (half2*)&frag_b1_3[1];

       half2* mat_h1_4 = (half2*)&frag_b0_4[0];
       half2* mat_h2_4 = (half2*)&frag_b0_4[1];
       half2* mat_h3_4 = (half2*)&frag_b1_4[0];
       half2* mat_h4_4 = (half2*)&frag_b1_4[1];
      
      sum += __half2float(vec_h1->x) * __half2float(mat_h1->x);
      sum += __half2float(vec_h1->y) * __half2float(mat_h1->y);
      sum += __half2float(vec_h2->x) * __half2float(mat_h2->x);
      sum += __half2float(vec_h2->y) * __half2float(mat_h2->y);
      sum += __half2float(vec_h3->x) * __half2float(mat_h3->x);
      sum += __half2float(vec_h3->y) * __half2float(mat_h3->y);
      sum += __half2float(vec_h4->x) * __half2float(mat_h4->x);
      sum += __half2float(vec_h4->y) * __half2float(mat_h4->y);

      sum_2 += __half2float(vec_h1->x) * __half2float(mat_h1_2->x);
      sum_2 += __half2float(vec_h1->y) * __half2float(mat_h1_2->y);
      sum_2 += __half2float(vec_h2->x) * __half2float(mat_h2_2->x);
      sum_2 += __half2float(vec_h2->y) * __half2float(mat_h2_2->y);
      sum_2 += __half2float(vec_h3->x) * __half2float(mat_h3_2->x);
      sum_2 += __half2float(vec_h3->y) * __half2float(mat_h3_2->y);
      sum_2 += __half2float(vec_h4->x) * __half2float(mat_h4_2->x);
      sum_2 += __half2float(vec_h4->y) * __half2float(mat_h4_2->y);
   
      
      sum_3 += __half2float(vec_h1->x) * __half2float(mat_h1_3->x);
      sum_3 += __half2float(vec_h1->y) * __half2float(mat_h1_3->y);
      sum_3 += __half2float(vec_h2->x) * __half2float(mat_h2_3->x);
      sum_3 += __half2float(vec_h2->y) * __half2float(mat_h2_3->y);
      sum_3 += __half2float(vec_h3->x) * __half2float(mat_h3_3->x);
      sum_3 += __half2float(vec_h3->y) * __half2float(mat_h3_3->y);
      sum_3 += __half2float(vec_h4->x) * __half2float(mat_h4_3->x);
      sum_3 += __half2float(vec_h4->y) * __half2float(mat_h4_3->y);

      sum_4 += __half2float(vec_h1->x) * __half2float(mat_h1_4->x);
      sum_4 += __half2float(vec_h1->y) * __half2float(mat_h1_4->y);
      sum_4 += __half2float(vec_h2->x) * __half2float(mat_h2_4->x);
      sum_4 += __half2float(vec_h2->y) * __half2float(mat_h2_4->y);
      sum_4 += __half2float(vec_h3->x) * __half2float(mat_h3_4->x);
      sum_4 += __half2float(vec_h3->y) * __half2float(mat_h3_4->y);
      sum_4 += __half2float(vec_h4->x) * __half2float(mat_h4_4->x);
      sum_4 += __half2float(vec_h4->y) * __half2float(mat_h4_4->y);

    }
  }

  // 计算outliers
  float* mat2_weight_fp16 = reinterpret_cast<float*>(weight);
  // half* vec2_outliers = reinterpret_cast<half*>(vec);
  // int * ind_outlier =  ind;

#pragma unroll
  for (int iter = 0; iter < num_outliers_per_thread >> 1; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n_outliers >> 1) {
      // float4 vec_val = vec4[j];
      float * mat_val = mat2_weight_fp16 +  row * (n_outliers >> 1) + j * 4;
      half2* read_mat = reinterpret_cast<half2*>(mat_val);

      int ind1 = ind[ 2 * j];
      int ind2 = ind[ 2 * j + 1];


      half vec_h1 = vec[ind1];
      half vec_h2 = vec[ind2];

        
      half2 mat_h1 = read_mat[0];
      half2 mat_h1_2 = read_mat[1];
      half2 mat_h1_3 = read_mat[2];
      half2 mat_h1_4 = read_mat[3];
     
      sum += __half2float(vec_h1) * __half2float(mat_h1.x);
      sum += __half2float(vec_h2) * __half2float(mat_h1.y);

      sum_2 += __half2float(vec_h1) * __half2float(mat_h1_2.x);
      sum_2 += __half2float(vec_h2) * __half2float(mat_h1_2.y);   

      sum_3 += __half2float(vec_h1) * __half2float(mat_h1_3.x);
      sum_3 += __half2float(vec_h2) * __half2float(mat_h1_3.y); 

      sum_4 += __half2float(vec_h1) * __half2float(mat_h1_4.x);
      sum_4 += __half2float(vec_h2) * __half2float(mat_h1_4.y);   
    }
  }

  sum = warpReduceSum(sum, blockDim.x);
  sum_2 = warpReduceSum(sum_2, blockDim.x);


  sum_3 = warpReduceSum(sum_3, blockDim.x);
  sum_4 = warpReduceSum(sum_4, blockDim.x);
 


  
  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2half(sum  * scaling[row]);
      res[row+1] =  __float2half(sum_2  * scaling[row + 1]);
      res[row+2] =  __float2half(sum_3  * scaling[row + 2]);
      res[row+3] = __float2half(sum_4  * scaling[row + 3]);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  static __shared__ float warpLevelSums_2[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  static __shared__ float warpLevelSums_3[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  static __shared__ float warpLevelSums_4[SHARED_MEM_MAX_ROWS][WARP_SIZE];

  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) {
      warpLevelSums[threadIdx.y][warpId] = sum;
      warpLevelSums_2[threadIdx.y][warpId] = sum_2;

      warpLevelSums_3[threadIdx.y][warpId] = sum_3;
      warpLevelSums_4[threadIdx.y][warpId] = sum_4;
  }
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;

  sum_2 = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums_2[threadIdx.y][laneId]
            : 0.0;

    sum_3 = (threadIdx.x < blockDim.x / WARP_SIZE)
  ? warpLevelSums_3[threadIdx.y][laneId]
  : 0.0;
    sum_4 = (threadIdx.x < blockDim.x / WARP_SIZE)
  ? warpLevelSums_4[threadIdx.y][laneId]
  : 0.0;
  // Final reduce using first warp
  if (warpId == 0) {
      sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
      sum_2 = warpReduceSum(sum_2, blockDim.x / WARP_SIZE);
      sum_3 = warpReduceSum(sum_3, blockDim.x / WARP_SIZE);
      sum_4 = warpReduceSum(sum_4, blockDim.x / WARP_SIZE);
  }
 
  if (tid == 0) {
      res[row] = __float2half(sum  * scaling[row]);
      res[row+1] = __float2half(sum_2  * scaling[row + 1]);

      res[row+2] = __float2half(sum_3  * scaling[row + 2]);
      res[row+3] = __float2half(sum_4  * scaling[row + 3]);

      //       res[row] = 1.0; //__float2half(sum  * scaling[row]);
      // res[row+1] = 2.0; // __float2half(sum_2  * scaling[row + 1]);
      // res[row+2] =3.0; // __float2half(sum_3  * scaling[row + 2]);
      // res[row+3] = 4.0; //__float2half(sum_4  * scaling[row + 3]);
  }
}

// add outliers
void gemv_int4_fp16_mix_sm90_cu_new(int m, int n, int k, half * vec_data_,
                                  void * mat_data_, 
                                  half * result_data_,
                                  unsigned int block_dim_x,
                                   unsigned int block_dim_y,
                                   float * scaling_data_,
                                    half * weight_cache_data,
                                    int * ind_data,
                                    int n_outliers,
                                   cudaStream_t stream
                                   ){
                      
 
  assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
  assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
  unsigned int num_per_thread = k / block_dim_x; 
  unsigned int num_outliers_per_thread = n_outliers / block_dim_x; 
  // 对于int4 block dim 应该减半
  assert(num_per_thread >= 8);
  dim3 grid_dim(1, (n / 4) / block_dim_y);
  // 需要的 block的数量减少一半
  dim3 block_dim(block_dim_x, block_dim_y);
  gemv_int4_fp16_kernel_sm90_new<<<grid_dim, block_dim, 1024, stream>>>((int32_t*) mat_data_, vec_data_, result_data_,
                                     k, num_per_thread, scaling_data_,
                                     weight_cache_data, ind_data, n_outliers, num_outliers_per_thread);
  checkCudaErrors(cudaPeekAtLastError());
  return  ;
}
