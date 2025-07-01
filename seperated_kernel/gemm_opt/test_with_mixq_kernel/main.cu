#include <cuda.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include "cutlass/cutlass.h"
#include "utils.cuh"
#include <iostream>

#include "symmetric/gemm/device/gemm_dequant.h"

template<int size>
__global__ void FindRowScaleKernel_(int8_t * output, const half * d_in, half * scale, int rows, int cols){

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
        scale[bid] = quant_scales;
    }
    // quant
    for (int i = tid ; i < cols; i += size)
        d_out[i] =  static_cast<int8_t>(__half2int_rn( __hdiv( start[i], quant_scales ) ))  ; 
    __syncthreads();    

}

void int8quant_(int rows, int cols, 
        const cutlass::half_t  * src, int8_t *output, 
        cutlass::half_t  *scale){


    dim3 block(256);
    dim3 grid(rows, 1);
    FindRowScaleKernel_<256><<<grid, block, 1024>>>(
                output,
                (half*) src, 
                (half*) scale,
                rows, cols);

}

void gemmfp16_(
    const half * mat1,
    const half * mat2, half *mat3, int m, int n, int k, cublasHandle_t handle) {
 

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

  

}
void int8FusedDequantizeCUDA_(const int8_t *A,
                             const int8_t *B,
                             const cutlass::half_t *scale_row,
                             const cutlass::half_t *scale_col,
                             cutlass::half_t *y, cutlass::half_t *D, 
                             int M, int N, int K,
                             char * workspace) {

 
  using Gemm = cutlass::gemm::device::symmetric::GemmDequant<
      int8_t,                          // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      int8_t,                          // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      cutlass::half_t,                 // ElementOutput
      cutlass::layout::RowMajor,       // LayoutOutput
      int32_t,                         // ElementAccumulator
      cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
      cutlass::arch::Sm80  // tag indicating target GPU compute architecture
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
      {(const cutlass::half_t *)scale_col, N},
      {(const cutlass::half_t *)scale_row, M},
      Gemm::ElementC(1)};

//   gemmOp.initialize(arguments, workspace);
    auto status = gemmOp(arguments);
//   gemmOp.run();
 
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
        start[ i * k ] = 0.0;
    }
 
 


}
void ExtractOutliersAndSetToZeros_(int M, int N, const half * A, half *fp_A, 
        const int *ind, const int len){


    const int blockSize = 128;
 

    half * tmp = const_cast<half*>(A);
    dim3 numBlocks(len);        
    FindOutliersAndSetToZeros_kernel_<<<numBlocks, blockSize, 1024>>>(
            ind,
            tmp,
            fp_A,
            M,
            N,
            len
        );

}
void test(const int m, const int n, const int k) {

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Failed to create cuBLAS handle" << std::endl;
    exit(0);
    }
    cudaEvent_t start, stop;
    float elapsedTime;
    int loop = 300;


    cutlass::half_t   *C_cublas_ptr, *A_cublas_ptr, *B_cublas_ptr , *B_quant_ptr, *A_quant_ptr;
    cudaMalloc(&C_cublas_ptr, sizeof(cutlass::half_t) * m * n);
    cudaMalloc(&A_cublas_ptr, sizeof(cutlass::half_t) * m * k);
    cudaMalloc(&B_cublas_ptr, sizeof(cutlass::half_t) * n * k);
    cudaMalloc(&A_quant_ptr, sizeof(cutlass::half_t) * m * k);
    cudaMalloc(&B_quant_ptr, sizeof(cutlass::half_t) * n * k);

    cutlass::half_t * hostA_cublas_ptr = (cutlass::half_t *)malloc(sizeof(cutlass::half_t) * m * k);
    cutlass::half_t * hostB_cublas_ptr = (cutlass::half_t *)malloc(sizeof(cutlass::half_t) * n * k);
    cutlass::half_t * hostC_cublas_ptr = (cutlass::half_t *)malloc(sizeof(cutlass::half_t) * m * n);
    gen_rand_data<cutlass::half_t>(hostA_cublas_ptr, m * k);
    gen_rand_data<cutlass::half_t>(hostB_cublas_ptr, n * k);



    cudaMemcpy(A_cublas_ptr, hostA_cublas_ptr, sizeof(cutlass::half_t) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(B_cublas_ptr, hostB_cublas_ptr, sizeof(cutlass::half_t) * n * k, cudaMemcpyHostToDevice);





     //---------------------------------------------------------------------------------------------------
    cutlass::half_t   *C_half_out_ptr, *tmp_ptr;
    int8_t  *int8_out_A, *int8_out_B;
    int8_t  *int8_debug_ptr  = (int8_t *)malloc(sizeof(int8_t) * k * n * 4);
    cutlass::half_t   *scale_a, *scale_b;
    cudaMalloc(&C_half_out_ptr, sizeof(cutlass::half_t) * m * n);
    cudaMalloc(&tmp_ptr, sizeof(cutlass::half_t) * m * n);
    cudaMalloc(&int8_out_A, sizeof(int8_t) * m * k);
    cudaMalloc(&int8_out_B, sizeof(int8_t) * n * k);

    cudaMalloc(&scale_a, sizeof(cutlass::half_t) * m);
    cudaMalloc(&scale_b, sizeof(cutlass::half_t) * n);

    const int num_ind = 128;
    int8quant_(m, k, A_cublas_ptr, int8_out_A, scale_a);
    int8quant_(n, k, B_cublas_ptr, int8_out_B, scale_b);



    cudaDeviceSynchronize();


    // printf("half input---------------\n");
    // cudaMemcpy(hostC_cublas_ptr, A_cublas_ptr, sizeof(cutlass::half_t) * n * k, cudaMemcpyDeviceToHost);
    // for (int i = 0 ; i < 20; ++i) {
    //     float tmp = (float) hostC_cublas_ptr[i];
    //     printf("%.4f\t", tmp);
    // }
    // printf("scale a output---------------\n");
    // cudaMemcpy(hostC_cublas_ptr, scale_a, sizeof(cutlass::half_t) * m , cudaMemcpyDeviceToHost);
    // for (int i = 0 ; i < 20; ++i) {
    //     float tmp = (float) hostC_cublas_ptr[i];
    //     printf("%.4f\t", tmp);
    // }

    // return ;
    // cudaMemcpy(hostC_cublas_ptr, scale_b, sizeof(cutlass::half_t) * n, cudaMemcpyDeviceToHost);
    // for (int i = 0 ; i < 20; ++i) {
    //     float tmp = (float) hostC_cublas_ptr[i];
    //     printf("%.4f\t", tmp);
    // }
    // printf("b scales---------------\n");
    // cudaMemcpy(int8_debug_ptr, int8_out_A, sizeof(int8_t) * m * k, cudaMemcpyDeviceToHost);
    // for (int i = 0 ; i < 20; ++i) {
    //     float tmp = (float) int8_debug_ptr[i];
    //     printf("%.4f\t", tmp);
    // }
    // printf("int 8 out---------------\n");
    // cudaMemcpy(int8_debug_ptr, int8_out_B, sizeof(int8_t) * n * k, cudaMemcpyDeviceToHost);
    // for (int i = 0 ; i < 20; ++i) {
    //     float tmp = (float) int8_debug_ptr[i];
    //     printf("%.4f\t", tmp);
    // }
     
    // cudaDeviceSynchronize();
    // return;



    

    // cutlass::half_t   *fp_activation, *fp_weight;
    // cudaMalloc(&fp_activation, sizeof(cutlass::half_t) * m * num_ind);
    // cudaMalloc(&fp_weight, sizeof(cutlass::half_t) * n * num_ind);


    // int *ind_cpu =   (int *)malloc(sizeof(int) * num_ind);
    // for (int i = 0 ; i < num_ind; ++i) ind_cpu[i] = i;
    // int   *ind ;
    // cudaMalloc(&ind, sizeof(int) *  num_ind);
    // cudaMemcpy(ind, ind_cpu, sizeof(ind) * num_ind, cudaMemcpyHostToDevice);

    // ExtractOutliersAndSetToZeros_(m, k, ( half *)A_quant_ptr, ( half *)fp_activation, ind, num_ind);
    // ExtractOutliersAndSetToZeros_(n, k, ( half *)B_quant_ptr, ( half *)fp_weight, ind, num_ind);
    // int8quant_(m, k, A_quant_ptr, int8_out_A, scale_a);
    // int8quant_(n, k, B_quant_ptr, int8_out_B, scale_b);

    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // // cublas
    // for (int i = 0; i < 10; ++i){
    //     gemmfp16_((half *)fp_activation, (half *)fp_weight, (half *)C_half_out_ptr, m, n, num_ind, handle);
    //     int8FusedDequantizeCUDA_(int8_out_A, int8_out_B, 
    //                         scale_a,
    //                         scale_b, 
    //                         C_half_out_ptr, 
    //                         C_half_out_ptr, 
    //                         m, n, k, 
    //                         reinterpret_cast<char*>(tmp_ptr));

    // }
 
    // cudaEventRecord(start, 0);
    // for (int i = 0; i < loop; ++i){
    //     gemmfp16_((half *)fp_activation, (half *)fp_weight, (half *)C_half_out_ptr, m, n, num_ind, handle);
    //     int8FusedDequantizeCUDA_(int8_out_A, int8_out_B, 
    //                         scale_a,
    //                         scale_b, 
    //                         C_half_out_ptr, 
    //                         C_half_out_ptr, 
    //                         m, n, k, 
    //                         reinterpret_cast<char*>(tmp_ptr));

    // }
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // printf("int8gemm + outliers FLOPs: %5.3fT\n", (2*float(n)*float(m)*float(k)/elapsedTime)/1e9 * loop);
    // cudaDeviceSynchronize();
    // ---------------------------------------------------------------------


    
    cutlass::half_t alpha = cutlass::half_t(1.0f);
    cutlass::half_t beta = cutlass::half_t(0.0f);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // cublas
    for (int i = 0; i < 10; ++i)
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                    &reinterpret_cast<const __half&>(alpha), (__half*)B_cublas_ptr, k, (__half*)A_cublas_ptr,
                     k, &reinterpret_cast<const __half&>(beta), (__half*)C_cublas_ptr, n);
 
    cudaEventRecord(start, 0);
    for (int i = 0; i < loop; ++i)
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                    &reinterpret_cast<const __half&>(alpha), (__half*)B_cublas_ptr, k, (__half*)A_cublas_ptr,
                     k, &reinterpret_cast<const __half&>(beta), (__half*)C_cublas_ptr, n);
        

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("%5.3fT,\t", (2*float(n)*float(m)*float(k)/elapsedTime)/1e9 * loop);
    cudaDeviceSynchronize();


    cudaMemcpy(hostC_cublas_ptr, C_cublas_ptr, sizeof(cutlass::half_t) * m * n, cudaMemcpyDeviceToHost);
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // cublas
    for (int i = 0; i < 10; ++i)
        int8FusedDequantizeCUDA_(int8_out_A, int8_out_B, 
                            scale_a,
                            scale_b, 
                            C_half_out_ptr, 
                            C_half_out_ptr, 
                            m, n, k, 
                            reinterpret_cast<char*>(tmp_ptr));
 
    cudaEventRecord(start, 0);
    for (int i = 0; i < loop; ++i)
        int8FusedDequantizeCUDA_(int8_out_A, int8_out_B, 
                            scale_a,
                            scale_b, 
                            C_half_out_ptr, 
                            C_half_out_ptr, 
                            m, n, k, 
                            reinterpret_cast<char*>(tmp_ptr));
        

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("%5.3fT,\t", (2*float(n)*float(m)*float(k)/elapsedTime)/1e9 * loop);
    cudaDeviceSynchronize();

   

    cudaMemcpy(A_quant_ptr, hostA_cublas_ptr, sizeof(cutlass::half_t) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(B_quant_ptr, hostB_cublas_ptr, sizeof(cutlass::half_t) * n * k, cudaMemcpyHostToDevice);

    cutlass::half_t   *fp_activation, *fp_weight;
    cudaMalloc(&fp_activation, sizeof(cutlass::half_t) * m * num_ind);
    cudaMalloc(&fp_weight, sizeof(cutlass::half_t) * n * num_ind);


    int *ind_cpu =   (int *)malloc(sizeof(int) * num_ind);
    for (int i = 0 ; i < num_ind; ++i) ind_cpu[i] = i;
    int   *ind ;
    cudaMalloc(&ind, sizeof(int) *  num_ind);
    cudaMemcpy(ind, ind_cpu, sizeof(int) * num_ind, cudaMemcpyHostToDevice);

    ExtractOutliersAndSetToZeros_(m, k, ( half *)A_quant_ptr, ( half *)fp_activation, ind, num_ind);
    ExtractOutliersAndSetToZeros_(n, k, ( half *)B_quant_ptr, ( half *)fp_weight, ind, num_ind);
    int8quant_(m, k, A_quant_ptr, int8_out_A, scale_a);
    int8quant_(n, k, B_quant_ptr, int8_out_B, scale_b);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 10; ++i){
        gemmfp16_((half *)fp_activation, (half *)fp_weight, (half *)C_half_out_ptr, m, n, num_ind, handle);
        int8FusedDequantizeCUDA_(int8_out_A, int8_out_B, 
                            scale_a,
                            scale_b, 
                            C_half_out_ptr, 
                            C_half_out_ptr, 
                            m, n, k, 
                            reinterpret_cast<char*>(tmp_ptr));

    }
 
    cudaEventRecord(start, 0);
    for (int i = 0; i < loop; ++i){
        gemmfp16_((half *)fp_activation, (half *)fp_weight, (half *)C_half_out_ptr, m, n, num_ind, handle);
        int8FusedDequantizeCUDA_(int8_out_A, int8_out_B, 
                            scale_a,
                            scale_b, 
                            C_half_out_ptr, 
                            C_half_out_ptr, 
                            m, n, k, 
                            reinterpret_cast<char*>(tmp_ptr));

    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("%5.3fT\n", (2*float(n)*float(m)*float(k)/elapsedTime)/1e9 * loop);
    cudaDeviceSynchronize();
    //---------------------------------------------------------------------

    if (m == 4096){
        printf("\n");
        for (int i = 0 ; i < 20; ++i) {
            float tmp = (float) hostC_cublas_ptr[i];
            printf("%.4f\t", tmp);
        }
        printf("---------------\n");


        cudaMemset(C_half_out_ptr, 0, sizeof(half) * m * n);
        gemmfp16_((half *)fp_activation, (half *)fp_weight, (half *)C_half_out_ptr, m, n, num_ind, handle);
        int8FusedDequantizeCUDA_(int8_out_A, int8_out_B, 
                            scale_a,
                            scale_b, 
                            C_half_out_ptr, 
                            C_half_out_ptr, 
                            m, n, k, 
                            reinterpret_cast<char*>(tmp_ptr));
        cudaMemcpy(hostC_cublas_ptr, C_half_out_ptr, sizeof(cutlass::half_t) * m * n, cudaMemcpyDeviceToHost);
        for (int i = 0 ; i < 20; ++i) {
            float tmp = (float) hostC_cublas_ptr[i];
            printf("%.4f\t", tmp);
        }
        printf("---------------\n");

    }

}
 

int main() {
    srand(10086);

    printf("cublas INT8,\tfused INT8,\tmixed INT8\n");
    const int n = 4096;
    const int k = 4096;
    
    // test(128,n,k);
    // test(256,n,k);
    test(512,n,k);
    test(1024,n,k);
    test(2048,n,k);
    test(4096,n,k);
    // test(8192,n,k);

    return 0;
}