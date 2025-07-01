from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.quantization.mixq import MixQLinearMethod

class MixQ4bitConfig(QuantizationConfig):
    """Config class for MixQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.pack_factor = 8
        

    def __repr__(self) -> str:
        return (f"MixQ4bitConfig(weight_bits={self.weight_bits}, ")

    def get_name(self) -> str:
        return "MixQ"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    def get_min_capability(self) -> int:
        # The MixQ kernel only supports Turing or newer GPUs.
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-MixQ
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-MixQ
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any])  :
        weight_bits = 4
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        # print("mix------weight")
        # print(weight_bits)
        # print(group_size)
        return cls(weight_bits, group_size)

    def get_quant_method(
            self, layer: torch.nn.Module,prefix: str
            ) -> Optional["MixQLinear4bitMethod"]:
        if isinstance(layer, LinearBase):
            #print("--get_quant_method---")
            # print(layer.prefix)
            if layer.prefix is not None and "down" in layer.prefix:
            # if 1:
                print("use 8bit!")
                return MixQLinearMethod(self)
            return MixQLinear4bitMethod(self)
    
        print(type(layer))
        # exit()
        return None

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]

import mixlib

import marlin
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           PackedvLLMParameter)
class MixQLinear4bitMethod(LinearMethodBase):


    def __init__(self, quant_config ):
        self.quant_config = quant_config
        self.debug = False


    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        output_size_per_partition = sum(output_partition_sizes)

 
        
        fp_features_num = 256
        weight_cache = Parameter(
            torch.empty(
                output_size_per_partition,
                fp_features_num,
                dtype=torch.float16,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight_cache, {
                "input_dim": 1,
                "output_dim": 0,
                 
            })
        layer.register_parameter("weight_cache", weight_cache)
        set_weight_attrs(weight_cache, extra_weight_attrs)
        

        ind = Parameter(
            torch.empty(
                fp_features_num,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("ind", ind)
        set_weight_attrs(ind, extra_weight_attrs)



        #  in_features // 16, out_features * 2
        # q_weight = Parameter(
        #     torch.zeros(
        #         input_size_per_partition // (self.quant_config.pack_factor * 2 ),
        #         output_size_per_partition * 2,
        #         dtype=torch.int32,
        #     ),
        #     requires_grad=False,
        # )
        # set_weight_attrs(
        #     q_weight, {
        #         "input_dim": 0,
        #         "output_dim": 1,
        #     })

        tile_size = 16
        pack_factor = 8
        weight_loader = extra_weight_attrs["weight_loader"]
        q_weight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // tile_size,
                output_size_per_partition * tile_size //
                pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor = pack_factor,
            # packed_factor=self.quant_config.pack_factor,
            marlin_tile_size= tile_size,
            weight_loader=weight_loader)
        q_weight.pack_factor = pack_factor

        q_scale_col = Parameter(
            torch.empty(
                input_size_per_partition // 128,
                output_size_per_partition,
                dtype=torch.float16,
            ),
            requires_grad=False,
        )
        set_weight_attrs(q_scale_col, {
            "input_dim": 0,
            "output_dim": 1,
        })

        layer.register_parameter("q_weight", q_weight)
        layer.register_parameter("q_scale_col", q_scale_col)
        # layer.register_parameter("qzeros", qzeros)
        # set_weight_attrs(q_weight, extra_weight_attrs)
        set_weight_attrs(q_scale_col, extra_weight_attrs)

        layer.init = False

        layer.output_size_per_partition = output_size_per_partition
        # layer.weight_cache2 = torch.clone(weight_cache.data)
        # layer.out = torch.zeros((1, output_size_per_partition), dtype=torch.half, device=weight_cache.data.device)
        # layer.w  = torch.zeros((output_size_per_partition, input_size_per_partition), 
        #     dtype = torch.float16, device = "cuda:0")
    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        
 

        inputs =  x.reshape(-1, x.shape[-1]) 
 
        M =  inputs.shape[0]
        N = layer.output_size_per_partition
        K = inputs.shape[1]


       
        thread_k = 64
        thread_n = 256
        y1 = torch.zeros((M, N), dtype=torch.half, device=inputs.device)
        workspace = torch.zeros(N // 128 * 16, device=inputs.device)
        marlin.mul(inputs, layer.q_weight, y1, layer.q_scale_col, workspace, thread_k, thread_n, 128)


        # print(inputs)
        # print(y1)
        # exit()
        # print(layer.q_scale_col)
        # ind = 8192 + 2048
        # print(layer.q_weight[0,:])
        # print(layer.q_weight.shape)
        # # print(ind)
        # exit()

        y1 = y1 + torch.mm(inputs[:,layer.ind], layer.weight_cache.data.T)
        if layer.bias is not None:
            y1 += layer.bias

        
        # print(y1)
        return y1
        
        
