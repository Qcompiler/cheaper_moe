from gen_data import gen_quant4, gen_quant4_my, gen_quant4_my_reshape
from gen_data import reshape_activation, reshape_weight
import torch
import mixgemm
import torch
import numpy as np
import torch.nn as nn


seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

DEV = torch.device('cuda:0')
k = 1024 
n = 256 

import safetensors
a = safetensors.safe_open("/home/dataset/DeepSeek-V2-Lite-Chat/model-00001-of-000004.safetensors", framework="pt")
w = a.get_tensor("model.layers.0.self_attn.q_proj.weight").cuda().half()
n, k = w.shape
# w = generate_mixed_matrix(n, k).cuda()
# print(matrix)
# w = torch.rand((n, k), dtype=torch.half, device=DEV) 

groupsize = 128
bit = 4
q_weight, scales  = gen_quant4_my(n, k, torch.clone(w),   groupsize = groupsize, tile = 1)
# print(q_weight)
# q_weight2, scales2  = gen_quant4_my(n, k, torch.clone(w),   groupsize = 128, tile = 1)
# print(q_weight2)
# print(q_weight2 - q_weight)
# 
# print(scales)
# print(scales2.shape)
# exit()

# scales = scales.t().contiguous()
print(scales.shape)
# exit()
scales = scales.to(DEV, dtype=torch.float32)
weight = mixgemm.dequant(q_weight, scales, n, k, bit, groupsize, 32, 4)  

print((weight - w).abs().mean())
# print(weight)
# print(w)