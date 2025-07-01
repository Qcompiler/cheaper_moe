from gen_data import gen_quant4, gen_quant4_my
from gen_data import reshape_activation, reshape_weight

import cutlass_int4_bf16_gemm
import torch
import mixgemm
import torch
import numpy as np
import torch.nn as nn
seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)


from EETQ import quant_weights, preprocess_weights, w8_a16_gemm
DEV = torch.device('cuda:0')
m = 1
# k = 11008
# n = 4096

# k = 256
# n = 256
n_outliers = 128
# q 5120
# k 1024
# v 1024
shapes = [(5120 + 1024 + 1024, 5120),  (5120, 5120), (27648, 5120), (5120, 27648)]

shapes = [ (4096, 4096)]

all_shape = []
for shape in shapes:

        k = shape[1]
        n = shape[0]
        A = torch.randn((m, k), dtype=torch.half, device=DEV)
        
        C = torch.zeros((m, n), dtype=torch.half, device=DEV) 
        
        w = torch.randn((n, k), dtype=torch.half, device=DEV) /100

        grand_fp16 = torch.mm(A,w.T)

        
        int8_weight_cpu = torch.t(w).contiguous().cpu()
        q_weight, q_scale_col = quant_weights(int8_weight_cpu, torch.int8, False)
        q_weight = q_weight.cuda()
        q_scale_col = q_scale_col.cuda()
        

        _, B_128, s_128 = gen_quant4(k, n, w.t().contiguous(),   groupsize = 128)

        B_, s_ = gen_quant4_my(n, k, w,   groupsize=-1, tile = 1)
        _, B, s = gen_quant4(n, k, w,   groupsize=-1)

        s_  = s_.to(torch.float32)
        C_i4_grand = torch.zeros((m, n), dtype=torch.half, device=DEV)
        mixgemm.gemv_int4(m, n, k, A, B_, C_i4_grand, 32, 4, s_.to(torch.float32))


        ind = torch.as_tensor(range(n_outliers)).to(torch.int32).cuda()
        weight_cache_ =   w[:,ind].contiguous()
        w[:,ind] = 0


        B_, s_ = gen_quant4_my(n, k, w,   groupsize=-1, tile = 1)
        weight_cache = weight_cache_ / s_.T
        s_  = s_.to(torch.float32)
        C_i4 = torch.zeros((m, n), dtype=torch.half, device=DEV)

        # print(B_[0:4,0:4])
        # print(s_[0:3])
        # exit()
        mixgemm.gemv_int4_fp16_mix(m, n, k, A, B_, C_i4, 32, 4, s_.to(torch.float32),weight_cache, ind, n_outliers)


        C_i4_grand = torch.zeros((m, n), dtype=torch.half, device=DEV)
        mixgemm.gemv_int4(m, n, k, A, B_, C_i4_grand, 32, 4, s_.to(torch.float32))

        C_i4_grand += torch.mm(A[:,ind], weight_cache_.T)





        mixgemm.gemv(m, n, k, A, w, C, 32, 4)

        import marlin
        workspace = torch.zeros(n // 128 * 16, device=DEV)
        C_i4mar = torch.zeros((m, n), dtype=torch.half, device=DEV)
        C_i4mar_g128 = torch.zeros((m, n), dtype=torch.half, device=DEV)
        thread_k = 64
        thread_n = 256
        marlin.mul(A, B, C_i4mar, s, workspace, thread_k, thread_n, -1)

        marlin.mul(A, B_128, C_i4mar_g128, s_128, workspace, thread_k, thread_n, -1)
        debug = True
        if debug:
                
                all = []

                for i in range(100):
                        # torch.mm(A, w.T)
                        mixgemm.gemv(m, n, k, A, w, C, 64, 4)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for i in range(100):
                        # torch.mm(A, w.T)
                        mixgemm.gemv(m, n, k, A, w, C, 64, 4)
                end_event.record()
                torch.cuda.synchronize()
                ms1 =  start_event.elapsed_time(end_event)
                start_event.record()
                # print("torch")
                # print(ms1)
                all.append(ms1)        
     

                for i in range(100):
                        marlin.mul(A, B_128, C_i4mar, s_128, workspace, thread_k, thread_n, -1)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for i in range(100):
                        marlin.mul(A, B_128, C_i4mar, s_128, workspace, thread_k, thread_n, -1)
                end_event.record()
                torch.cuda.synchronize()
                ms1 =  start_event.elapsed_time(end_event)
                start_event.record()
                # print("int4 marlin g 128")
                # print(ms1)

                all.append(ms1)        

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for i in range(100):
                        mixgemm.gemv_int4_fp16_mix(m, n, k, A, B, C_i4, 32, 4, s_, weight_cache, ind, n_outliers)
                end_event.record()
                torch.cuda.synchronize()
                ms1 =  start_event.elapsed_time(end_event)
                start_event.record()
                # print("int4 mix")
                # print(ms1)

                all.append(ms1)
                weight_cache  =  reshape_weight(weight_cache)
                # exit()
                B  =  reshape_activation(B)

                for i in range(100):
                        mixgemm.gemv_int4_fp16_mix_sm90(m, n, k, A, B, C_i4, 64, 8, s_, weight_cache, ind, n_outliers)

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for i in range(100):
                        mixgemm.gemv_int4_fp16_mix_sm90(m, n, k, A, B, C_i4, 64, 8, s_, weight_cache, ind, n_outliers)
                end_event.record()
                torch.cuda.synchronize()
                ms1 =  start_event.elapsed_time(end_event)
                start_event.record()
                # print("int4 mix sm90")
                # print(ms1)

                all.append(ms1)






                for i in range(100):
                        y =  w8_a16_gemm(A, q_weight, q_scale_col)

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for i in range(100):
                        y =  w8_a16_gemm(A, q_weight, q_scale_col)
                end_event.record()
                torch.cuda.synchronize()
                ms1 =  start_event.elapsed_time(end_event)
                start_event.record()

                all.append(ms1)

                s_f32 =  s_.to(torch.float32)
                for i in range(100):
                        mixgemm.gemv_int4_sm90(m, n, k, A, B_, C_i4_grand, 32, 4, s_f32)


                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for i in range(100):
                        mixgemm.gemv_int4_sm90(m, n, k, A, B_, C_i4_grand, 32, 4, s_f32)
                end_event.record()
                torch.cuda.synchronize()
                ms1 =  start_event.elapsed_time(end_event)
                start_event.record()

                all.append(ms1)


                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for i in range(100):
                        mixgemm.gemv_int4_sm90(m, n, k, A, B_, C_i4_grand, 64, 4, s_f32)
                end_event.record()
                torch.cuda.synchronize()
                ms1 =  start_event.elapsed_time(end_event)
                start_event.record()

                all.append(ms1)

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for i in range(100):
                        mixgemm.gemv_int4_sm90(m, n, k, A, B_, C_i4_grand, 128, 4, s_f32)
                end_event.record()
                torch.cuda.synchronize()
                ms1 =  start_event.elapsed_time(end_event)
                start_event.record()
                all.append(ms1)

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for i in range(100):
                        mixgemm.gemv_int4_sm90(m, n, k, A, B_, C_i4_grand, 32, 8, s_f32)
                end_event.record()
                torch.cuda.synchronize()
                ms1 =  start_event.elapsed_time(end_event)
                start_event.record()
                all.append(ms1)


                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for i in range(100):
                        mixgemm.gemv_int4_sm90(m, n, k, A, B_, C_i4_grand, 64, 8, s_f32)
                end_event.record()
                torch.cuda.synchronize()
                ms1 =  start_event.elapsed_time(end_event)
                start_event.record()
                all.append(ms1)



                a = torch.randn(n, k // 8, dtype=torch.float16, device='cuda').to(torch.int32)
                b = torch.randn(m, k, dtype=torch.float16, device='cuda')
                
                c = torch.zeros(m, n, dtype=torch.float16, device='cuda')
                # Call the CUTLASS GEMM function
                
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for i in range(100):
                        d = cutlass_int4_bf16_gemm.int4_bf16_gemm_convert_only(b, a, c, 1.0, 0.0)
                end_event.record()
                torch.cuda.synchronize()
                ms1 =  start_event.elapsed_time(end_event)
                start_event.record()
                all.append(ms1)


                all_shape.append(all)

import pandas as pd



df = (pd.DataFrame(all_shape,columns=["fp16", "marlin",  "mix", "mixsm90", "INT8", 
                                      "INT4 Peak (32 4)", "INT4 Peak (64 4)", 
                                      "INT4 Peak (128 4)", "INT4 Peak (32 8)", "INT4 Peak (64 8)", "55 Hopper INT4"], 
                   index = ["qkv", "dense", "gate", "down"]))

print(df)
import matplotlib.pyplot as plt


# 1. 绘制原始数据的柱状图并保存
plt.figure(figsize=(10, 5))
df.plot(kind='bar')
plt.title('Original Data')
plt.ylabel('Values')
plt.tight_layout()  # 自动调整布局防止标签被截断
plt.savefig('original_data.png', dpi=300, bbox_inches='tight')  # 保存为PNG
plt.show()

# 2. 将所有列除以第一列进行归一化
# 注意：确保第一列没有零值
normalized_df = 1./df.div(df.iloc[:, 0], axis=0)

# 3. 绘制归一化后的柱状图并保存
plt.figure(figsize=(10, 5))
normalized_df.plot(kind='bar')
# plt.title('不同shape 在H100上的加速比')
plt.ylabel('Speed Up Ratio')
plt.tight_layout()  # 自动调整布局防止标签被截断
plt.savefig('normalized_data.png', dpi=300, bbox_inches='tight')  # 保存为PNG
plt.show()