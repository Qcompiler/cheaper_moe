 
import torch
import torch.nn as nn
import sys
import mixlib
# import marlin
 
import mixgemm

# from EETQ import quant_weights, preprocess_weights, w8_a16_gemm

from torch import Tensor

from vllm import _custom_ops as ops
# import vllm._C

def two_compl(x: Tensor, bits: int) -> Tensor:
    return torch.where(x < 0, 2 ** bits + x, x)
def pack_to_i4(X: Tensor):

    X_i8 = two_compl(X.to(dtype=torch.int8), 4).to(torch.uint8)
    X_i4 = X_i8[:, 0::2] | (X_i8[:, 1::2] << 4)
    return X_i4

def unpack_int8_to_int4(weight,ind):
    assert weight.dim() == 2
    return mixlib.unpack_int4_to_fp16(weight,ind)
 

NN = 1024
class MixLinear_GEMM(nn.Module):
    def __init__(self, in_features, out_features, bias, dev,  bit, 
            weight_only = False, cache = None, fp_features_num = 256):
        super().__init__()
        
 
        self.in_features = in_features
        self.out_features = out_features
        self.bit = bit
        print("current bit is ", bit)
        print(in_features)
        print(out_features)

        self.fp_features_num = fp_features_num


        self.register_buffer('weight_cache', torch.empty((out_features, fp_features_num),
                                                                    device=dev,
                                                                    dtype=torch.float16, 
                                                                    requires_grad=False))
        self.register_buffer('ind', torch.empty(
            (fp_features_num), dtype=torch.int32,device=dev, requires_grad=False)) 


        if bit == 8:
            self.register_buffer('q_weight', torch.empty((out_features, in_features), 
                                dtype=torch.int8, device=dev,requires_grad=False))
            self.register_buffer('q_scale_col', torch.empty((1, out_features), 
                        dtype=torch.float16, device=dev,requires_grad=False))
        else:
            self.register_buffer('q_weight', torch.empty(( out_features, in_features // 8), 
                                                            dtype=torch.int,
                                                              device=dev,requires_grad=False))
            self.register_buffer('q_scale_col', torch.empty((out_features, in_features // 128),
                                                             dtype=torch.float16, 
                                                             device=dev,requires_grad=False))



        if bias:

            self.register_buffer('bias', torch.empty((out_features), dtype=torch.float16, device=dev,requires_grad=False))
        else:
            self.bias = None
        self.cnt = 0
        self.forward_without_precondition_len = 128

        self.cache = cache
        self.weight_only = weight_only


        self.add_outliers = False

         
        if cache is not None:
            self.sigma = torch.ones((1, 1),dtype=torch.float16, requires_grad=False,
                                            device = dev)
            self.sigma[0] = cache.sigma

        self.arch = torch.cuda.get_device_capability()[0]
        self.init = False
        self.w = None

        self.fp8 = True




    @classmethod
    def from_linear(cls, linear, bit, fp_features_num , weight_only=False, init_only=False,cache=None,                     
                    layer_scales= None, dev = 'cuda'):


        quant_linear = cls(linear.in_features, linear.out_features, linear.bias is not None, 
                           dev, bit=bit, weight_only=weight_only,
                           cache=cache, fp_features_num = fp_features_num)
   
        
        if init_only is True: 
            return quant_linear   


        
        
        print("current bit is ", bit)

        assert layer_scales is not None
        fp_features = quant_linear.fp_features_num
        linear.ind = torch.sort(layer_scales)[1][-fp_features:]


        tmp = torch.clone(linear.weight.data.cuda())               
        quant_linear.weight_cache.copy_(tmp[:, linear.ind].to(tmp.dtype).cuda())  
        quant_linear.ind.copy_(linear.ind.cuda().to(torch.int32))

        # quant_linear.new_ind.copy_(torch.sort(layer_scales)[1][:NN].cuda().to(torch.int32))
        # quant_linear.new_fp_weight.copy_(tmp[:, quant_linear.new_ind].to(tmp.dtype).cuda())  
        
        if bit == 8:

            scale =   (torch.max(torch.abs(tmp), dim=1)[0].unsqueeze(1) / (
                        127)).to(torch.float16).reshape((1,linear.out_features))
        else:
            tmp[:, linear.ind] = 0
            # scale =   (torch.max(torch.abs(tmp), dim=1)[0].unsqueeze(1) / (10)).to(torch.float16).reshape((1,linear.out_features))
        
            # print(scale)
            # exit()
        # quant_linear.scale_col.copy_(scale)
        
        
        if bit == 4:
            from .mixed4bit import gen_quant4_my, gen_quant4

            
            w = torch.clone(tmp.cuda())
            # w[:, linear.ind] = 0
            n, k = w.shape
            B, s = gen_quant4_my(n, k, w,  groupsize = 128, tile = 1)
            # _, B, s = gen_quant4(k, n, w.t().contiguous(),   groupsize=128) 

            quant_linear.q_weight.copy_ (B.cpu())
            quant_linear.q_scale_col.copy_(s.half().cpu()) 
             
        if bit == 8:
            tmp /= scale.T
            tmp = tmp.round().to(torch.int8)
            quant_linear.q_weight.copy_ (tmp)
            quant_linear.q_scale_col.copy_(scale.half())
 
        if linear.bias is not None:
            quant_linear.bias.copy_(linear.bias.half())

        return quant_linear
    
 
         

    
    @torch.no_grad()
    def FindOutliers(self,Activation):

        
        tmp = torch.unique(torch.where((  Activation.abs() > self.sigma ))[1])
        return tmp.to(torch.int32)


    @torch.no_grad()
    def forward(self, x,   unfused = True):

        inputs =  (x.reshape(-1, x.shape[-1]))

        if inputs.shape[0] == 0:
            return  torch.zeros((0, self.out_features), device=x.device, dtype=x.dtype)

        # print(inputs.shape)
        shape = x.shape[:-1] + (self.out_features, )
        if self.bit == 4:
     
            
    
            M =  inputs.shape[0]
            N = self.out_features
            K = inputs.shape[1]

            if self.init is False:
                self.init = True
                # print("init int4")
                # self.w = self.q_weight.to(torch.float16) * self.q_scale_col.to(torch.float16).T
                
                self.scales = self.q_scale_col.to(inputs.device, dtype=torch.float32)
                self.weight = mixgemm.dequant(self.q_weight, self.scales, N, K, 4, 128, 32, 4)  
                self.q_weight = self.q_weight.cpu()
                if self.fp8 is not True:
                    pass
                else:    
                    
                    self.q_weight_fp8 = torch.zeros(self.out_features, self.in_features, device = inputs.device, 
                                                    dtype=torch.float8_e4m3fn)
                    self.scale_weight_fp8 = torch.ones((1), device=inputs.device, dtype=torch.float32)
                    torch.ops._C.dynamic_scaled_fp8_quant(self.q_weight_fp8, self.weight, self.scale_weight_fp8)
                    self.weight = self.weight.cpu()

                torch.cuda.empty_cache()
                # print(torch.cuda.memory_allocated())

            if self.fp8 is not True:
                y1 = torch.mm(inputs, self.weight.T)
                y1 = y1 + torch.mm(inputs[:,self.ind], self.weight_cache.data.T)

                if self.bias is not None:
                    y1 += self.bias
            else:
                import mixgemm_v2

                
                tmp = inputs
                fp8input = torch.empty_like(tmp, dtype=torch.float8_e4m3fn)
                scale_input = torch.ones((1), device=inputs.device, dtype=torch.float32)
                torch.ops._C.dynamic_scaled_fp8_quant(fp8input, tmp, scale_input)
                bs = 1
                seq = inputs.shape[0]
                assert  fp8input.shape[0] > 0
                y1 =   mixgemm_v2.cutlass_scaled_mm_fp8(bs, seq, 
                                            self.out_features, 
                                            self.in_features,
                                            fp8input,
                                            self.q_weight_fp8.T,
                                            scale_input,
                                            self.scale_weight_fp8,
                                            self.bias, 2)
                outliers = torch.mm(inputs[:,self.ind], self.weight_cache.data.T)
                y1 = y1 + outliers

            
            

        if self.bit == 8:
 
            M =  inputs.shape[0]
            N = self.out_features
            K = inputs.shape[1]

                
            if self.init is False:
                self.init = True
                if self.fp8 is not True:
                    self.w = self.q_weight.to(torch.float16) * self.q_scale_col.to(torch.float16).T
                    self.q_weight = self.q_weight.cpu()
                
                else:
                    tmp = self.q_weight.to(torch.float16) * self.q_scale_col.to(torch.float16).T
                    self.q_weight = self.q_weight.cpu()
                    self.q_weight_fp8 = torch.zeros(self.out_features, self.in_features, device = inputs.device, 
                                                    dtype=torch.float8_e4m3fn)
                    self.scale_weight_fp8 = torch.ones((1), device=inputs.device, dtype=torch.float32)
                    torch.ops._C.dynamic_scaled_fp8_quant(self.q_weight_fp8, tmp, self.scale_weight_fp8)
                    tmp = tmp.cpu()

                torch.cuda.empty_cache()
                # print(torch.cuda.memory_allocated())
            
            if self.fp8 is not True:

                y1 = torch.mm(inputs, self.w.T)
                if self.bias is not None:
                    y1 += self.bias
            else:
                import mixgemm_v2

                # print(inputs.shape)
                fp8input = torch.empty_like(inputs, dtype=torch.float8_e4m3fn)
                scale_input = torch.ones((1), device=inputs.device, dtype=torch.float32)
                torch.ops._C.dynamic_scaled_fp8_quant(fp8input, inputs, scale_input)
                bs = 1
                seq = inputs.shape[0]

                # print(fp8input)
                assert  fp8input.shape[0] > 0
                y1 =   mixgemm_v2.cutlass_scaled_mm_fp8(bs, seq, 
                                            self.out_features, 
                                            self.in_features,
                                            fp8input,
                                            self.q_weight_fp8.T,
                                            scale_input,
                                            self.scale_weight_fp8,
                                            self.bias, 2)

            
        assert  y1.shape[0] > 0
        return y1.reshape(shape)


 