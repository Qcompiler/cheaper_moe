from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs


# arch = torch.cuda.get_device_capability()
# arch = arch[0] * 10 + arch[1] 

class MixQConfig(QuantizationConfig):
    """Config class for MixQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int
    ) -> None:
        self.weight_bits = weight_bits
        self.pack_factor = 8
        # self.group_size = group_size
        

    def __repr__(self) -> str:
        return (f"MixQConfig(weight_bits={self.weight_bits}, ")

    def get_name(self) -> str:
        return "MixQ"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    def get_min_capability(self) -> int:
        # The MixQ kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-MixQ
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-MixQ
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MixQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        # group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        #print("mix------weight")
        #print(weight_bits)
        #print(group_size)
        return cls(weight_bits)

    def get_quant_method(
            self, layer: torch.nn.Module, prefix: str) -> Optional["MixQLinearMethod"]:
        if isinstance(layer, LinearBase):
            # print("--get_quant_method---")
            # print(layer.prefix)
            if layer.prefix is not None and "down" in layer.prefix:
                return MixQLinearMethod(self, weight_only = True)
            return MixQLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]

import mixlib

class MixQLinearMethod(LinearMethodBase):


    def __init__(self, quant_config: MixQConfig, weight_only = False):
        self.quant_config = quant_config
        self.weight_only = weight_only
        self.use_exact = False
        


    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):

        output_size_per_partition = sum(output_partition_sizes)



        q_weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            q_weight, {
                "input_dim": 1,
                "output_dim": 0,
                
            })
        q_scale_col = Parameter(
            torch.empty(
                1,
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
        set_weight_attrs(q_weight, extra_weight_attrs)


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


        layer.register_parameter("q_scale_col", q_scale_col)
        set_weight_attrs(q_scale_col, extra_weight_attrs)


        layer.output_size_per_partition = output_size_per_partition
        layer.w = None
        layer.init = False
    def apply_(self, layer, x, bias):
        shape = x.shape[:-1] + (layer.output_size_per_partition, )


        inputs = x.reshape(-1, x.shape[-1])
        M =  inputs.shape[0]
        N = layer.output_size_per_partition
        K = inputs.shape[1]

            
        if layer.init is False:
            layer.init = True
            layer.w = layer.q_weight.to(torch.float16) * layer.q_scale_col.to(torch.float16).T

        # print(w.shape)
        # print(layer.q_weight.shape)
        # print(layer.q_scale_col)

        # print(layer.q_weight)
        # print(w)
        # exit()
        y1 = torch.mm(inputs, layer.w.T)



        if layer.bias is not None:
            y1 += layer.bias
        
        return y1.reshape(shape)
        

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        return self.apply_(layer, x, bias)
 
