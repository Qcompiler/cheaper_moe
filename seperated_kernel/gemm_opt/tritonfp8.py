
import torch
import flashinfer
from vllm import _custom_ops as ops


m = 4096
n = 4096
k = 4096
device = 'cuda'
out_dtype = torch.bfloat16
def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)    
a = to_fp8(torch.randn((m, k), device=device))
b = to_fp8(torch.randn((n, k), device=device).t())

per_token_act_quant = False
per_out_channel_weight_quant = False
m_a_scales = m if per_token_act_quant else 1
n_b_scales = n if per_out_channel_weight_quant else 1

scale_a = (torch.randn((m_a_scales, 1), device=device,
                        dtype=torch.float32))
scale_b = (torch.randn((1, n_b_scales), device=device,
                        dtype=torch.float32))

bias = None
out = ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

loop = 300

st = torch.cuda.Event(True)
ed = torch.cuda.Event(True)

for _ in range(10):
    out = ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

st.record()
for _ in range(loop):
    #flashinfer.gemm.bmm_fp8(A_fp8, B_fp8, A_scale, B_scale, torch.bfloat16, output)
    out = ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

ed.record()
ed.synchronize()
print(((m * n * k * 2) * loop / st.elapsed_time(ed)) / 1e9)

