from .base import BaseForCausalLM
from typing import Dict
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

class LlamaMixQForCausalLM(BaseForCausalLM):
    layer_type = "LlamaDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"


    @staticmethod
    def get_model_layers(model: LlamaForCausalLM):
        return model.model.layers
    
 
    @staticmethod
    def move_embed(model: LlamaForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
  
