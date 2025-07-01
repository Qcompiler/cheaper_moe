# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Callable, Tuple
import flashinfer
import click

import torch
import triton  # @manual

import mixgemm_v2
 
from torch._tensor import Tensor
from vllm import _custom_ops as ops
import json
@click.command()
@click.option("--cuda-graph", type=bool, default=True)
def bench(cuda_graph: bool) -> None:
    """Benchmark bf16 vs scale/cast + int8."""
    dic = {}
    def _run_benchmark(
        bench_factory: Callable[
            [torch.Tensor, torch.Tensor], Callable[[], torch.Tensor]
        ],
        shape: Tuple[int, int, int] = (1024, 1024, 1024),
        tag: str = "",
    ) -> None:
        if tag not in dic.keys():
            dic[tag] = []
        # Benchmarks the function returned by bench_factory.
        # Any pre-processing that should not be benchmarked can occur inside bench_factory.
        m, n, k = shape

        input_shape = (m, k)
        weight_shape = (n, k)

        base_dtype = torch.bfloat16
        input_ = torch.randn(input_shape, device="cuda", dtype=base_dtype)
        weight_ = torch.randn(weight_shape, device="cuda", dtype=base_dtype)

        gemm_fn = bench_factory(input_, weight_)

        if cuda_graph:
            bench_stream = torch.cuda.Stream()
            with torch.cuda.stream(bench_stream):
                ms = triton.testing.do_bench_cudagraph(
                    lambda: gemm_fn(),
                    rep=100,
                )
        else:
            ms = triton.testing.do_bench(
                lambda: gemm_fn(),
                warmup=25,
                rep=100,
            )

        tflops = (2 * m * n * k) / 1e12
        sec = ms / 1e3
        perf_str = f"{tflops / sec:.2f}"
        
        dic[tag].append([perf_str,ms])
        #print(dic)
        print(
            f"{(tag + ':').ljust(40)}\tshape {str(shape):<25} tflops {perf_str:<8} ms {ms:.3f}"
        )

    shapes = [
        (m, 10240, 8192) for m in [32,64,128,256,512,1024,2048,4096,8192]
    ]
    # shapes = [
    #     (m, 4096, 4096) for m in [32]
    # ]
    for shape in shapes:
        #_run_benchmark(mixed_cutlass_scaled_mm_bench, shape=shape, tag="mixed_cutlass_scaled_mm_bench")
        _run_benchmark(bf16_bench, shape=shape, tag="bf16")
        # _run_benchmark(int8_scaled_mm_bench, shape=shape, tag="int8_scaled_mm_bench int8")
        _run_benchmark(cutlass_scaled_mm_bench, shape=shape, tag="cutlass_scaled_mm_bench")
        print("------------------------------------")
        

    with open("save.json","w", encoding='utf-8') as f:  
        f.write(    json.dumps(   dic  ,ensure_ascii=False     )     ) 

        

def bf16_bench(x: Tensor, w: Tensor) -> Callable[[], Tensor]:
    def gemm_fn() -> Tensor:
        return torch.matmul(x, w.T)

    return gemm_fn
def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(
        min=-128, max=127)).to(dtype=torch.int8)    
def cutlass_scaled_mm_bench(x: Tensor, w: Tensor) -> Callable[[], Tensor]:

    device = 'cuda'
    out_dtype = torch.bfloat16
 
    a = to_int8(x)
    b = to_int8(w)

    per_token_act_quant = True
    per_out_channel_weight_quant = True
    m_a_scales = x.shape[0] if per_token_act_quant else 1
    n_b_scales = w.shape[0] if per_out_channel_weight_quant else 1

    scale_a = (torch.randn((m_a_scales, 1), device=device,
                            dtype=torch.float32))
    scale_b = (torch.randn((1, n_b_scales), device=device,
                            dtype=torch.float32))

    bias = None
   

    out = torch.zeros((x.shape[0],w.shape[0]),device=device,dtype=out_dtype)
    
    def gemm_fn() -> Tensor:

        mixgemm_v2.cutlass_scaled_mm(out, a, b.t(), scale_a, scale_b, None)  
        return out
        return   ops.cutlass_scaled_mm(a, b.t(), scale_a, scale_b, out_dtype, bias)

    return gemm_fn

def int8_scaled_mm_bench(x: Tensor, w: Tensor) -> Callable[[], Tensor]:

    device = 'cuda'
    out_dtype = torch.float16
 
    a = to_int8(x)
    b = to_int8(w)

    per_token_act_quant = True
    per_out_channel_weight_quant = True
    m_a_scales = x.shape[0]  
    n_b_scales = w.shape[0]  

    scale_a = (torch.randn((m_a_scales, 1), device=device,
                            dtype=torch.float16))
    scale_b = (torch.randn((1, n_b_scales), device=device,
                            dtype=torch.float16))

    bias = None
    

    out = torch.zeros((x.shape[0], w.shape[0]),device=device,dtype=out_dtype)
    def gemm_fn() -> Tensor:

 
        return  mixgemm.int8FusedDequantize(a, b, scale_a, scale_b,
                                             out, m_a_scales, n_b_scales,  x.shape[1] )   

    return gemm_fn

def mixed_cutlass_scaled_mm_bench(x: Tensor, w: Tensor) -> Callable[[], Tensor]:

    device = 'cuda'
    out_dtype = torch.bfloat16

    a = to_int8(x)
    b = to_int8(w).t()

    per_token_act_quant = True
    per_out_channel_weight_quant = True
    m_a_scales = x.shape[0] if per_token_act_quant else 1
    n_b_scales = w.shape[0] if per_out_channel_weight_quant else 1

    scale_a = (torch.randn((m_a_scales, 1), device=device,
                            dtype=torch.float32))
    scale_b = (torch.randn((1, n_b_scales), device=device,
                            dtype=torch.float32))

    bias = None
   
    outliers_A = (torch.randn((x.shape[0], 32), device=device,
                            dtype=out_dtype))
    outliers_B = (torch.randn((w.shape[0], 32), device=device,
                            dtype=out_dtype))

    out = torch.zeros((x.shape[0],w.shape[0]),device=device,dtype=out_dtype)
    def gemm_fn() -> Tensor:
        # void cutlass_scaled_mm(torch::Tensor& out, torch::Tensor const& a,
        #                torch::Tensor const& b, torch::Tensor const& a_scales,
        #                torch::Tensor const& b_scales,
        #                c10::optional<torch::Tensor> const& bias);
        mixgemm.cutlass_scaled_mm(out, a, b, scale_a, scale_b,None)
        return   out + torch.mm(outliers_A, outliers_B.T)


    return gemm_fn




def cublas_int8(x: Tensor, w: Tensor) -> Callable[[], Tensor]:

    x = x.reshape(1,x.shape[0],x.shape[1])
    w = w.reshape(1,w.shape[0],w.shape[1])
    A_scale = x.abs().max() / 127
    A_int8 = (x / A_scale).to(torch.int8)
    B_scale = w.abs().max() / 127
    B_int8 = (w / B_scale).to(torch.int8)

    A_scale = A_scale.to(torch.float32)

    B_scale = B_scale.to(torch.float32)

    output = torch.empty((1, x.shape[1], w.shape[1]), device='cuda', dtype=torch.bfloat16)

    def run_gemm() -> Tensor:
        return flashinfer.gemm.bmm_int8(A_int8, B_int8, A_scale, B_scale,
         torch.bfloat16, output)

    return run_gemm



if __name__ == "__main__":
    dic = bench()
    print("-------")
    print(dic)


