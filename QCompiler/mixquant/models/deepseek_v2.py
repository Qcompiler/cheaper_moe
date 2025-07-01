from .base import BaseForCausalLM
from typing import Dict
import torch
from typing import List, Tuple, Union
from mixquant.utils.utils import set_module_name
from mixquant.modules.fused.mlp import  MixQwen2MLP
from mixquant.modules.fused.attn import QuantAttentionFused
from mixquant.modules.fused.norm import FasterTransformerRMSNorm
from mixquant.modules.linear import  MixLinear_GEMM


import sys



class deepseek_v2MixQForCausalLM(BaseForCausalLM):
    layer_type = "Qwen2DecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model, quant_config: Dict, mix = False, cache = None):
 
        pass

 
    @staticmethod
    def get_model_layers(model):
        return model.model.layers
    
 
    
    @staticmethod
    def move_embed(model, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
  
