import torch
import flashinfer
m = 4096
n = 4096
k = 4096

A = torch.randn((1, 4096, 4096), device='cuda', dtype=torch.float32)
A_scale = A.abs().max() / 448
A_fp8 = (A / A_scale).to(torch.float8_e4m3fn)
B = torch.randn((1, 4096, 4096), device='cuda', dtype=torch.float32)
B_scale = B.abs().max() / 448
B_fp8 = (B / B_scale).to(torch.float8_e4m3fn)

output = torch.empty((1, 4096, 4096), device='cuda', dtype=torch.bfloat16)
loop = 300

st = torch.cuda.Event(True)
ed = torch.cuda.Event(True)

for _ in range(10):
    flashinfer.gemm.bmm_fp8(A_fp8, B_fp8, A_scale, B_scale, torch.bfloat16, output)

st.record()

for _ in range(loop):
    flashinfer.gemm.bmm_fp8(A_fp8, B_fp8, A_scale, B_scale, torch.bfloat16, output)

ed.record()
ed.synchronize()
print(((m * n * k * 2) * loop / st.elapsed_time(ed)) / 1e9)
