#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <torch/all.h>
// #include "cutlass/cutlass.h"

// #include "scaled_mm_c2x.cuh"
// #include "scaled_mm_c2x_sm89_fp8_dispatch.cuh"
// #include "scaled_mm_c2x_sm89_int8_dispatch.cuh"
#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>

// template <template <typename, typename> typename Epilogue,
//           typename... EpilogueArgs>
// void cutlass_scaled_mm_sm89_epilogue(torch::Tensor& out, torch::Tensor const& a,
//                                      torch::Tensor const& b,
//                                      EpilogueArgs&&... epilogue_args) {
//   if (a.dtype() == torch::kInt8) {
//     TORCH_CHECK(b.dtype() == torch::kInt8);

//     if (out.dtype() == torch::kBFloat16) {
//       return vllm::cutlass_gemm_sm89_int8_dispatch<int8_t, cutlass::bfloat16_t,
//                                                    Epilogue>(
//           out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
//     } else {
//       assert(out.dtype() == torch::kFloat16);
//       return vllm::cutlass_gemm_sm89_int8_dispatch<int8_t, cutlass::half_t,
//                                                    Epilogue>(
//           out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
//     }
//   } else {
//     TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
//     TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

//     if (out.dtype() == torch::kBFloat16) {
//       return vllm::cutlass_gemm_sm89_fp8_dispatch<
//           cutlass::float_e4m3_t, cutlass::bfloat16_t, Epilogue>(
//           out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
//     } else {
//       TORCH_CHECK(out.dtype() == torch::kFloat16);
//       return vllm::cutlass_gemm_sm89_fp8_dispatch<cutlass::float_e4m3_t,
//                                                   cutlass::half_t, Epilogue>(
//           out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
//     }
//   }
// }

// void cutlass_scaled_mm_sm89(torch::Tensor& out, torch::Tensor const& a,
//                             torch::Tensor const& b,
//                             torch::Tensor const& a_scales,
//                             torch::Tensor const& b_scales,
//                             c10::optional<torch::Tensor> const& bias) {
//   TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
//   TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
//   if (bias) {
//     TORCH_CHECK(bias->dtype() == out.dtype(),
//                 "currently bias dtype must match output dtype ", out.dtype());
//     return cutlass_scaled_mm_sm89_epilogue<vllm::ScaledEpilogueBias>(
//         out, a, b, a_scales, b_scales, *bias);
//   } else {
//     return cutlass_scaled_mm_sm89_epilogue<vllm::ScaledEpilogue>(
//         out, a, b, a_scales, b_scales);
//   }
// }
// int32_t get_sm_version_num() {
//     int32_t major_capability, minor_capability;
//     cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
//                             0);
//     cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
//                             0);
//     int32_t version_num = major_capability * 10 + minor_capability;
//     return version_num;
// }
// void cutlass_scaled_mm_sm89(torch::Tensor& c, torch::Tensor const& a,
//                             torch::Tensor const& b,
//                             torch::Tensor const& a_scales,
//                             torch::Tensor const& b_scales,
//                             c10::optional<torch::Tensor> const& bias);
// void cutlass_scaled_mm_sm90(torch::Tensor& c, torch::Tensor const& a,
//                             torch::Tensor const& b,
//                             torch::Tensor const& a_scales,
//                             torch::Tensor const& b_scales,
//                             c10::optional<torch::Tensor> const& bias);
// void cutlass_scaled_mm(torch::Tensor& c, torch::Tensor const& a,
//                        torch::Tensor const& b, torch::Tensor const& a_scales,
//                        torch::Tensor const& b_scales,
//                        c10::optional<torch::Tensor> const& bias) {
//   // Checks for conformality
//   TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
//   TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
//               b.size(1) == c.size(1));
//   TORCH_CHECK(a_scales.numel() == 1 || a_scales.numel() == a.size(0));
//   TORCH_CHECK(b_scales.numel() == 1 || b_scales.numel() == b.size(1));

//   // Check for strides and alignment
//   TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
//   TORCH_CHECK(b.stride(0) == 1);                      // Column-major
//   TORCH_CHECK(c.stride(0) % 16 == 0 &&
//               b.stride(1) % 16 == 0);  // 16 Byte Alignment
//   TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

//   if (bias) {
//     TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous() &&
//                 bias->dim() == 1);
//   }

//   at::cuda::OptionalCUDAGuard const device_guard(device_of(a));
//   int32_t version_num = get_sm_version_num();

//   if (version_num == 89) {
//     // Ada Lovelace
//     cutlass_scaled_mm_sm89(c, a, b, a_scales, b_scales, bias);
//     return;
//   }else{
 
//     if (version_num >= 90) {
//         cutlass_scaled_mm_sm90(c, a, b, a_scales, b_scales, bias);
//         return;
//       }

 
//     else{
//       TORCH_CHECK_NOT_IMPLEMENTED(
//       false,
//       "No compiled cutlass_scaled_mm for a compute capability less than "
//       "CUDA device capability: ",
//       version_num);
//     }
//   }





// }
