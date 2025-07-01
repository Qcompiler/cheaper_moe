from gen_data import gen_quant4, gen_quant4_my, gen_quant4_my_reshape
from gen_data import reshape_activation, reshape_weight

from gen_data import reshape_activation_new, reshape_weight_new


import torch
import mixgemm
import torch
import numpy as np
import torch.nn as nn
seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

DEV = torch.device('cuda:0')
m = 1
# k = 11008
# n = 4096
k = 4096 
n = 4096 
# k = 256
# n = 256
n_outliers = 128


A = torch.randn((m, k), dtype=torch.half, device=DEV)

C = torch.zeros((m, n), dtype=torch.half, device=DEV) 



w = torch.randn((n, k), dtype=torch.half, device=DEV) /100
ind = torch.as_tensor(range(n_outliers)).to(torch.int32).cuda()


grand_fp16 = torch.mm(A,w.T)


import mixgemm

# _, B_, s_ = gen_quant4_my(n, k, w,   groupsize=-1, tile = 1)



# extract outliers 
def __get_outliers(w, ind):
    w_ = torch.clone(w)
    
    weight_cache_ =   w_[:,ind].contiguous()
    # weight_cache_ *= 0
    w_[:,ind] = 0
    _, B_set_zeros, s_set_zeros = gen_quant4_my(n, k, w_,   groupsize=-1, tile = 1)
    weight_cache = weight_cache_ / s_set_zeros.T
    s_set_zeros  = s_set_zeros.to(torch.float32)

    return B_set_zeros, s_set_zeros, weight_cache



weight_cache_ =   w[:,ind].contiguous() 

qweight, scales = gen_quant4_my(n, k, w,   groupsize=-1, tile = 1)

weight_cache =  (weight_cache_ / scales.T ) .to(torch.float16)

C_i4 = torch.zeros((m, n), dtype=torch.half, device=DEV)
mixgemm.gemv_int4_fp16_mix(m, n, k, A, qweight, C_i4, 64, 4, scales.to(torch.float32), 
                        weight_cache, ind, n_outliers)


# weight_cache *= 0
# fp16
weight_cache_  =  reshape_weight(weight_cache)


# qweight
B_set_zeros_  =  reshape_activation(qweight)


C_i4_reshaped = torch.zeros((m, n), dtype=torch.half, device=DEV)
mixgemm.gemv_int4_fp16_mix_sm90(m, n, k, A, B_set_zeros_, C_i4_reshaped, 64, 4, scales.to(torch.float32), 
                        weight_cache_, ind, n_outliers)


weight_cache_new  =  reshape_weight_new(weight_cache)
B_set_zeros_new  =  reshape_activation_new(qweight)
C_i4_reshaped_new = torch.zeros((m, n), dtype=torch.half, device=DEV)


mixgemm.gemv_int4_fp16_mix_sm90_new(m, n, k, A, B_set_zeros_new, C_i4_reshaped_new, 64, 4, scales.to(torch.float32), 
                        weight_cache_new, ind, n_outliers)



print(C_i4_reshaped)
print(C_i4_reshaped_new)
# print(C_i4)
# print(grand_fp16)