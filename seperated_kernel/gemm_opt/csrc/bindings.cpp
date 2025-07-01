#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>


torch::Tensor int8FusedDequantize(const torch::Tensor &A,
                                  const torch::Tensor &B,
                                  const torch::Tensor &scale_row,
                                  const torch::Tensor &scale_col,
                                  const torch::Tensor &y, int m, int n, int k);
torch::Tensor cutlass_scaled_mm( int batch, int seq_len,  int N, int K,
                       torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       c10::optional<torch::Tensor> const& bias);

torch::Tensor cutlass_scaled_mm_fp8( int batch, int seq_len,  int N, int K,
                       torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       c10::optional<torch::Tensor> const& bias, int dim );
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cutlass_scaled_mm", &cutlass_scaled_mm, "cutlass_scaled_mm ");
  m.def("cutlass_scaled_mm_fp8", &cutlass_scaled_mm_fp8, "cutlass_scaled_mm_fp8 ");
  m.def("int8FusedDequantize", &int8FusedDequantize, "int8FusedDequantize ");
}
