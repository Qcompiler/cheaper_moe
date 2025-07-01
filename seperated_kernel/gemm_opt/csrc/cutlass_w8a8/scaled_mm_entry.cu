#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

void cutlass_scaled_mm_sm75(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            c10::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_sm80(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            c10::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_sm89(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            c10::optional<torch::Tensor> const& bias);

#if defined ENABLE_SCALED_MM_C3X && ENABLE_SCALED_MM_C3X
void cutlass_scaled_mm_sm90(torch::Tensor& c, 
                            torch::Tensor const& a,
                            torch::Tensor const& b,
                            int m, int n, int k, 
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            c10::optional<torch::Tensor> const& bias);
#endif

void cutlass_scaled_mm_azp_sm75(torch::Tensor& c, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                c10::optional<torch::Tensor> const& azp,
                                c10::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_azp_sm80(torch::Tensor& c, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                c10::optional<torch::Tensor> const& azp,
                                c10::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_azp_sm89(torch::Tensor& c, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                c10::optional<torch::Tensor> const& azp,
                                c10::optional<torch::Tensor> const& bias);

#if defined CUDA_VERSION && CUDA_VERSION >= 12000
void cutlass_scaled_mm_azp_sm90(torch::Tensor& c, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                c10::optional<torch::Tensor> const& azp,
                                c10::optional<torch::Tensor> const& bias);
#endif

bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability) {
  // CUTLASS FP8 kernels need at least
  //   CUDA 12.0 on SM90 systems (Hopper)
  //   CUDA 12.4 on SM89 systems (Lovelace)

#if defined CUDA_VERSION
  if (cuda_device_capability >= 90) {
    return CUDA_VERSION >= 12000;
  } else if (cuda_device_capability >= 89) {
    return CUDA_VERSION >= 12040;
  }
#endif

  return false;
}

int32_t get_sm_version_num() {
  int32_t major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         0);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         0);
  int32_t version_num = major_capability * 10 + minor_capability;
  return version_num;
}



template<int size>
__global__ void FindRowScaleKernel_(int8_t * output, const half * d_in, float * scale, int rows, int cols){

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


    // write result for this block to global mem
    //if (tid < 32) warpReduce(sdata, tid);

    __syncthreads();
    float tmp = 127.0;
    half quant_scales = __hdiv( sdata[0], __float2half(tmp));
    if (tid == 0){
        scale[bid] = (__half2float)(quant_scales);
    }
    // quant
    for (int i = tid ; i < cols; i += size)
        d_out[i] =  static_cast<int8_t>(__half2int_rn( __hdiv( start[i], quant_scales ) ))  ; 
    __syncthreads();    

}
void int8quant_(int rows, int cols, const half * src, int8_t *output, 
        float *scale, cudaStream_t stream){


    dim3 block(256);
    dim3 grid(rows, 1);
    FindRowScaleKernel_<256><<<grid, block, 1024, stream>>>(
                output,
                src, scale,
                rows, cols);

};
torch::Tensor cutlass_scaled_mm( int batch, int seq_len,  int N, int K,
                       torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       c10::optional<torch::Tensor> const& bias) {
  // Checks for conformality

  // TORCH_CHECK(a_scales.numel() == 1 || a_scales.numel() == a.size(0));
  // TORCH_CHECK(b_scales.numel() == 1 || b_scales.numel() == b.size(1));

  // Check for strides and alignment
  // TORCH_CHECK(a.stride(1) == 1 );  // Row-major
  // TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  // TORCH_CHECK(b.stride(1) % 16 == 0);  // 16 Byte Alignment
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  if (bias) {
    TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous() &&
                bias->dim() == 1);
  }

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));
  int32_t version_num = get_sm_version_num();
  // Hopper
  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(a.device());  
  auto c = torch::zeros(
        {batch, seq_len,  N }, options);

  int M = batch * seq_len;

  auto options_i8 = torch::TensorOptions().dtype(torch::kInt8).device(a.device());
  auto quant_out = torch::zeros(
      { M, K }, options_i8);
  int8_t* int8_out = reinterpret_cast<int8_t *>(quant_out.data_ptr<int8_t>());
  float* scale_a = reinterpret_cast<float *>(a_scales.data_ptr<float>());
  const half * A = reinterpret_cast<half const*>(a.data_ptr<at::Half>());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 
  int8quant_(M, K, A, int8_out, scale_a, stream);
  if (version_num >= 90) {
    cutlass_scaled_mm_sm90(c, quant_out, b, M, N, K, a_scales, b_scales, bias);
    
  }

  return  c;

}



torch::Tensor cutlass_scaled_mm_fp8( int batch, int seq_len,  int N, int K,
                       torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       c10::optional<torch::Tensor> const& bias, int dim ) {

  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  if (bias) {
    TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous() &&
                bias->dim() == 1);
  }

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));
  int32_t version_num = get_sm_version_num();
  // Hopper
  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(a.device());  
  torch::Tensor c;

  if (dim == 3)
    c = torch::zeros(
        {batch, seq_len,  N }, options);

  if (dim == 2)
    c = torch::zeros(
        {batch * seq_len,  N }, options); 

  int M = batch * seq_len;

  if (version_num >= 90) {
    cutlass_scaled_mm_sm90(c, a, b, M, N, K, a_scales, b_scales, bias);
    
  }

  return  c;

}

void cutlass_scaled_mm_azp(torch::Tensor& c, torch::Tensor const& a,
                           torch::Tensor const& b,
                           torch::Tensor const& a_scales,
                           torch::Tensor const& b_scales,
                           torch::Tensor const& azp_adj,
                           c10::optional<torch::Tensor> const& azp,
                           c10::optional<torch::Tensor> const& bias) {
  // Checks for conformality
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
              b.size(1) == c.size(1));
  TORCH_CHECK(a_scales.numel() == 1 || a_scales.numel() == a.size(0));
  TORCH_CHECK(b_scales.numel() == 1 || b_scales.numel() == b.size(1));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  TORCH_CHECK(c.stride(0) % 16 == 0 &&
              b.stride(1) % 16 == 0);  // 16 Byte Alignment
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  // bias, azp, azp_adj are all 1d
  // bias and azp_adj have n elements, azp has m elements
  if (bias) {
    TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous());
  }
  if (azp) {
    TORCH_CHECK(azp->numel() == a.size(0) && azp->is_contiguous());
  }
  TORCH_CHECK(azp_adj.numel() == b.size(1) && azp_adj.is_contiguous());

  // azp & bias types
  TORCH_CHECK(azp_adj.dtype() == torch::kInt32);
  TORCH_CHECK(!azp || azp->dtype() == torch::kInt32);
  TORCH_CHECK(!bias || bias->dtype() == c.dtype(),
              "currently bias dtype must match output dtype ", c.dtype());

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));

  int32_t version_num = get_sm_version_num();

#if defined ENABLE_SCALED_MM_C3X && ENABLE_SCALED_MM_C3X
  if (version_num >= 90) {
    cutlass_scaled_mm_azp_sm90(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }
#endif

#if defined ENABLE_SCALED_MM_C2X && ENABLE_SCALED_MM_C2X
  if (version_num == 89) {
    // Ada Lovelace
    cutlass_scaled_mm_azp_sm89(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }

  if (version_num >= 80) {
    // Ampere
    cutlass_scaled_mm_azp_sm80(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }

  // Turing
  TORCH_CHECK(version_num >= 75);
  cutlass_scaled_mm_azp_sm75(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
  return;
#endif

  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_mm_azp for a compute capability less than "
      "CUDA device capability: ",
      version_num);
}