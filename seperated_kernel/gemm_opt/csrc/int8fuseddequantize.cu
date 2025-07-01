#include "symmetric/gemm/device/gemm_dequant.h"
// #include "symmetric/symmetric_internal.h"
#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <torch/all.h>
#include "cutlass/cutlass.h"
#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>

torch::Tensor int8FusedDequantizeCUDA(
                                      const torch::Tensor &A,
                                      const torch::Tensor &B,
                                      const torch::Tensor &scale_row,
                                      const torch::Tensor &scale_col,
                                      const torch::Tensor &y, int M, int N, int K) {
  torch::checkAllSameGPU("int8FusedDequantize", {{A, "A", 0},
                                                 {B, "B", 1},
                                                 {scale_row, "scale_row", 2},
                                                 {scale_col, "scale_col", 3},
                                                 {y, "y", 4}});

  auto D = torch::empty({M, N}, torch::dtype(torch::kF16).device(A.device()));
  auto work = torch::empty({M, N}, torch::dtype(torch::kF16).device(A.device()));  
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

  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {A.data_ptr<int8_t>(), K},
      {B.data_ptr<int8_t>(), K},
      {(cutlass::half_t *)y.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)D.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)scale_col.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)scale_row.data_ptr<torch::Half>(), M},
      Gemm::ElementC(1)};

  gemmOp.initialize(arguments, reinterpret_cast<char*>(work.data_ptr<torch::Half>()),
   at::cuda::getCurrentCUDAStream().stream());

  gemmOp.run(at::cuda::getCurrentCUDAStream().stream());

  return D;
}


torch::Tensor int8FusedDequantize(const torch::Tensor &A,
                                  const torch::Tensor &B,
                                  const torch::Tensor &scale_row,
                                  const torch::Tensor &scale_col,
                                  const torch::Tensor &y,  int M, int N, int K) {



                                    
  torch::checkAllContiguous("int8FusedDequantize", {{A, "A", 0},
                                                    {B, "B", 1},
                                                    {scale_row, "scale_row", 2},
                                                    {scale_col, "scale_col", 3},
                                                    {y, "y", 4}});
  torch::checkDeviceType("int8FusedDequantize", {A, B, scale_row, scale_col, y},
                         at::DeviceType::CUDA);
  return int8FusedDequantizeCUDA(A, B, scale_row, scale_col, y, M, N, K);
}
