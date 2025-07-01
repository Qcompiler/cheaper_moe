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
m = 1
# k = 11008
# n = 4096
k = 512 
n = 1024 
# k = 256
# n = 256
n_outliers = 128

def test_with_outliers(m,n,k):


    A = torch.randn((m, k), dtype=torch.half, device=DEV)
    
    C = torch.zeros((m, n), dtype=torch.half, device=DEV) 
    


    w = torch.randn((n, k), dtype=torch.half, device=DEV) /100
    ind = torch.as_tensor(range(n_outliers)).to(torch.int32).cuda()
    # w [:, ind ] = 0

    grand_fp16 = torch.mm(A,w.T)
    
    _, B_128, s_128 = gen_quant4(k, n, w.t().contiguous(),   groupsize = 128) 

    _, B, s = gen_quant4(k, n, w.t().contiguous(),   groupsize=-1)  

    import marlin
    workspace = torch.zeros(n // 128 * 16, device=DEV)
    C_i4mar = torch.zeros((m, n), dtype=torch.half, device=DEV)
    C_i4mar_g128 = torch.zeros((m, n), dtype=torch.half, device=DEV)
    thread_k = 64
    thread_n = 256
    marlin.mul(A, B, C_i4mar, s, workspace, thread_k, thread_n, -1)

    marlin.mul(A, B_128, C_i4mar_g128, s_128, workspace, thread_k, thread_n, -1)

    import mixgemm

    _, B_, s_ = gen_quant4_my(n, k, w,   groupsize=-1, tile = 1)
    _, B_sm90, s_sm90 = gen_quant4_my_reshape(n, k, w,   groupsize=-1, tile = 1)
    C_i4_my = torch.zeros((m, n), dtype=torch.half, device=DEV)
    mixgemm.gemv_int4(m, n, k, A, B_, C_i4_my, 32, 4, s_.to(torch.float32))

    C_i4_sm90 = torch.zeros((m, n), dtype=torch.half, device=DEV)
    mixgemm.gemv_int4_sm90(m, n, k, A, B_sm90, C_i4_sm90, 32, 4, s_sm90.to(torch.float32))
    
    

    
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
    
    



    B_set_zeros, s_set_zeros, weight_cache = __get_outliers(w, ind)
    C_i4 = torch.zeros((m, n), dtype=torch.half, device=DEV)
    mixgemm.gemv_int4_fp16_mix(m, n, k, A, B_set_zeros, C_i4, 64, 4, s_set_zeros.to(torch.float32), 
                           weight_cache, ind, n_outliers)
    

 
      
    print(weight_cache[0:8,0:8])
    # for sm90
    weight_cache  =  reshape_weight(weight_cache)
    print(weight_cache[0:8,0:8])
    # exit()
    B_set_zeros  =  reshape_activation(B_set_zeros)


    C_i4_reshaped = torch.zeros((m, n), dtype=torch.half, device=DEV)
    mixgemm.gemv_int4_fp16_mix_sm90(m, n, k, A, B_set_zeros, C_i4_reshaped, 64, 4, s_set_zeros.to(torch.float32), 
                           weight_cache, ind, n_outliers)
    
    
    # print("marlin")
    # print(C_i4mar)
    # print("marlin g 128")
    # print(C_i4mar_g128)
    print("my int4 gemm is ")
    print(C_i4_my)


    print("my int4 gemm _sm90 is ")
    print(C_i4_sm90)
    
    print("my gemm with outliers is ")
    print(C_i4)

    print("with outliers sm90")
    print(C_i4_reshaped)
    # print("grand is ")
    # print(grand_fp16)

    print("check errors of reshaped weight")
    print(  (C_i4_sm90 - C_i4_my ).abs().max())
    print("check errors of reshaped weight with mix")
    print(  (C_i4 - C_i4_reshaped ).abs().max())


    
test_with_outliers(m,n,k)
